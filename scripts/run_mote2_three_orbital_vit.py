"""ViT × Slater parent run script — MoTe2 three-orbital model.

Running this script creates a timestamped job folder under `jobs/`, copies
itself there as `run_script.py`, and executes VMC optimisation followed by
observable collection.  Re-running `run_script.py` from inside its own job
folder resumes an interrupted run (checkpoint loading is automatic).

Directory layout after one run
-------------------------------
results/mote2_three_orbital/vit/
├── run.py                          ← this file (never modified by a run)
└── jobs/
    └── vit_scm_3_0_0_3_nf3_V1_0.30_mc_YYYYMMDD_HHMMSS/
        ├── run_script.py           ← exact copy of run.py at launch time
        ├── run.log                 ← stdout + stderr
        ├── gpu_mem.log             ← per-step GPU memory
        ├── summary.json
        ├── vstate_variables.mpack  ← final checkpoint (full variables)
        ├── vstate_parameters_leaves.npz
        ├── vstate_parameters_treedef.txt
        ├── runtime_log.json
        ├── checkpoints/
        │   ├── step_500.mpack      ← 10 evenly-spaced intermediate checkpoints
        │   ├── step_1000.mpack
        │   └── ...
        ├── raw_data/
        │   ├── optimization_history.npz
        │   └── observables_data_step<N>.npz
        ├── plots_optimization/
        └── plots_observables/

Environment variables (all optional)
--------------------------------------
  CUDA_VISIBLE_DEVICES        GPU index (default: 0)
  V1                          Interaction strength (default: 0.3)
  SAMPLE_TYPE                 MC | FullSum (default: MC)
  N_ITER                      Optimisation steps (default: 5000)
  N_SAMPLES                   MC samples per step (default: 4096)
  N_DISCARD_PER_CHAIN         Discard per chain per step (default: 4)
  SWEEP_SIZE                  MCMC sweep size (default: 2*n_fermions)
  N_CHAINS                    Number of MCMC chains (default: 1024)
  LEARNING_RATE               SGD learning rate (default: 0.01)
  DIAG_SHIFT                  SR diagonal shift (default: 0.001)
  OBS_N_SAMPLES               Samples for post-training observables (default: 1 000 000)
  OBS_N_DISCARD_PER_CHAIN     Discard for observable sampling (default: max(64, N_DISCARD))
  CHUNK_SIZE                  MCState chunk size for grad + local_estimators (default: 32).
                              The ViT uses O(n_sites²) attention so the flat batch
                              n_samples × max_conn_size is very large for 81-site systems.
                              chunk_size=32 reduces the XLA batch to 32×max_conn per kernel.
                              Set to 0 to disable chunking (only safe for small systems).
  NUM_LAYERS                  ViT encoder depth (default: 2)
  D_MODEL                     Token embedding dimension (default: 32)
  N_HEADS                     Attention heads (default: 4)
  MLP_HIDDEN_FACTOR           MLP hidden-layer factor (default: 4)
  OUTPUT_HIDDEN_DIM           ViT output hidden dimension (default: 32)
  DISPLACEMENT_ONLY_ATTENTION 1 | 0. If 1, alpha_ij depends only on relative
                              displacement class. If 0, alpha_ij is independent
                              for every ordered pair (i, j). (default: 1)
  SLATER_INIT_MODE            random | noninteracting (default: random)

Validation test (non-interacting, small system)
-------------------------------------------------
  CUDA_VISIBLE_DEVICES=2 V1=0 SAMPLE_TYPE=FullSum N_ITER=500 \\
    SUPERCELL_MATRIX="2,0,0,2" N_FERMIONS=4 \\
    python results/mote2_three_orbital/vit/run.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# JAX memory pre-allocation note
# --------------------------------
# Do NOT default XLA_PYTHON_CLIENT_PREALLOCATE to "false" here.
# When preallocation is disabled, JAX's BFC allocator satisfies many small
# chunk allocations by growing the pool piecemeal; this fragments the heap
# and can prevent a subsequent large contiguous allocation (e.g. the SR
# Jacobian, ~1.6 GiB) from succeeding even when total free memory is >> that.
# Keeping the default behaviour (preallocate ~90% of GPU memory at startup)
# avoids fragmentation.  If you need to share the GPU, export
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 (or lower) before launching.

# Disable XLA GPU kernel autotuning.  On Ada Lovelace GPUs (RTX 6000 Ada,
# RTX 4090, …) the autotuner sometimes fails with "No valid config found!"
# for non-power-of-2 attention batch sizes (n_samples × max_conn_size).
# Setting level=0 skips benchmarking and uses the default kernel selection.
# Must be set before the first `import jax`.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_autotune_level" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_autotune_level=0").lstrip()

import flax.serialization
import jax

# Use this to get the actual GPU name
try:
    print(f"GPU name: {jax.lib.xla_bridge.get_backend().platform_version}")
except AttributeError:
    print(f"JAX devices: {jax.devices()}")

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import netket as nk
import numpy as np

from aidan_custom.bloch_ed import (
    build_fock_basis,
    build_mote2_three_orbital_lattice_embedding,
    build_mote2_three_orbital_projected_hamiltonian,
    build_mote2_three_orbital_real_space_terms,
    solve_projected_all_momentum_sectors,
)
from aidan_custom.geometry import fold_to_shortest_k
from aidan_custom.models import LogSlaterSpatialViT, make_translation_equivariant_pair_data
from aidan_custom.mote2_three_orbital import (
    MOTE2_A1,
    MOTE2_A2,
    mote2_three_orbital_reciprocal_vectors,
)
from aidan_custom.mote2_three_orbital_model import (
    build_mote2_three_orbital_hamiltonian,
    noninteracting_slater_orbitals_mote2_three_orbital,
)
from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
)
from aidan_custom.optimization import make_optimization_callback

_LAUNCH_ROOT = Path(__file__).resolve().parent.parent / "results" / "mote2_three_orbital" / "vit"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    return bool(int(os.environ.get(key, str(int(default)))))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _build_geometry(
    supercell_matrix: np.ndarray,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Return site positions, supercell vectors sc_t1/sc_t2, and lattice."""
    lattice = build_mote2_three_orbital_lattice_embedding(
        supercell_matrix=np.asarray(supercell_matrix, dtype=np.int64),
    )
    a1 = float(a_m) * np.asarray(MOTE2_A1, dtype=np.float64)
    a2 = float(a_m) * np.asarray(MOTE2_A2, dtype=np.float64)
    u0 = np.array([float(a_m) / np.sqrt(3.0), 0.0])
    offsets = np.array([u0, np.zeros(2), -u0])

    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell
    positions = np.empty((n_sites, 2))
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        bravais = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = (
            float(bravais[0]) * a1
            + float(bravais[1]) * a2
            + offsets[int(orb)]
        )

    sc_t1 = (
        float(lattice.supercell_matrix[0, 0]) * a1
        + float(lattice.supercell_matrix[1, 0]) * a2
    )
    sc_t2 = (
        float(lattice.supercell_matrix[0, 1]) * a1
        + float(lattice.supercell_matrix[1, 1]) * a2
    )
    return positions, sc_t1, sc_t2, lattice


def _fold_positions(
    positions: np.ndarray,
    sc_t1: np.ndarray,
    sc_t2: np.ndarray,
) -> np.ndarray:
    """Fold Cartesian positions into the supercell rhombus (frac coords in [0,1))."""
    sc_cols = np.column_stack([sc_t1, sc_t2])
    frac = np.linalg.solve(sc_cols, positions.T).T
    frac_folded = frac - np.floor(frac)
    return (sc_cols @ frac_folded.T).T


def _supercell_q_vectors(
    lattice,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Supercell BZ q-points folded to the MoTe2 primitive Wigner-Seitz BZ.

    Uses fold_to_shortest_k with the primitive MoTe2 reciprocal vectors b1, b2
    (not the supercell ones), so each raw supercell q is mapped to the shortest
    equivalent representative in the primitive reciprocal lattice, i.e. the
    first Brillouin zone (Wigner-Seitz cell of the reciprocal lattice).
    """
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    b1 = np.asarray(b1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    q_raw = np.array([
        lattice.kpoint_coefficients[kx, ky][0] * b1
        + lattice.kpoint_coefficients[kx, ky][1] * b2
        for kx in range(lattice.Lx)
        for ky in range(lattice.Ly)
    ])
    # fold_to_shortest_k returns the representative with minimum |q| — the WS-BZ.
    q_folded = np.array([
        fold_to_shortest_k(q, b1, b2, search_radius=8) for q in q_raw
    ])
    return q_raw, q_folded


def _hashable_matrix(arr: np.ndarray) -> tuple:
    arr = np.asarray(arr, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


# ---------------------------------------------------------------------------
# Pair data for ViT
# ---------------------------------------------------------------------------

def _make_pair_data(
    positions: np.ndarray,
    sc_t1: np.ndarray,
    sc_t2: np.ndarray,
) -> tuple[tuple, tuple, np.ndarray, np.ndarray]:
    """Compute translation-equivariant pair data for the ViT.

    The supercell IS the simulation cell, so Lx=Ly=1 with basis_vectors =
    [sc_t1, sc_t2].  The minimum-image translations are
        {n1*sc_t1 + n2*sc_t2 : n1,n2 ∈ {-1,0,1}}
    which gives correct PBC minimum-image displacements for the ViT attention.
    """
    sc_basis = np.stack([sc_t1, sc_t2], axis=0)  # (2,2), rows = supercell vectors
    pair_classes, pair_distances, _ = make_translation_equivariant_pair_data(
        positions=positions,
        basis_vectors=sc_basis,
        Lx=1,
        Ly=1,
        pbc=np.array([True, True]),
    )
    pc_np = np.asarray(pair_classes, dtype=np.int32)
    pd_np = np.asarray(pair_distances, dtype=np.float64)
    pc_hash = tuple(tuple(int(v) for v in row) for row in pc_np)
    pd_hash = tuple(float(v) for v in pd_np)
    return pc_hash, pd_hash, pc_np, pd_np


# ---------------------------------------------------------------------------
# Non-interacting reference energy
# ---------------------------------------------------------------------------

def _noninteracting_energy(
    supercell_matrix: np.ndarray,
    n_fermions: int,
    *,
    delta: float,
    ez: float,
    t_th1: float,
    t_hh1: float,
    t_th2: float,
    t_hh3: float,
    t_tt1: float,
    a_m: float,
    ph_conj: bool,
) -> float:
    """Sum of the n_fermions lowest one-body eigenvalues (V1=0)."""
    _, h1, _ = build_mote2_three_orbital_real_space_terms(
        supercell_matrix=supercell_matrix,
        delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
        t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
        a_m=a_m, ph_conj=ph_conj, V1=0.0,
    )
    evals = np.linalg.eigvalsh(np.asarray(h1, dtype=np.complex128))
    return float(np.real(evals[:n_fermions].sum()))


# ---------------------------------------------------------------------------
# Bloch-ED reference energies
# ---------------------------------------------------------------------------

_ED_COLORS = {1: "tab:orange", 2: "tab:green", 3: "tab:red"}
_ED_LS     = {1: "--",         2: "-.",         3: ":"}


def _compute_mote2_ed_reference_energies(
    supercell_matrix: np.ndarray,
    n_fermions: int,
    delta: float,
    ez: float,
    t_th1: float,
    t_hh1: float,
    t_th2: float,
    t_hh3: float,
    t_tt1: float,
    a_m: float,
    ph_conj: bool,
    V1: float,
    dim_cutoff: int = 100_000,
) -> dict[int, float | None]:
    """Bloch-ED ground-state energies projected into 1, 2, and 3 bands.

    Returns {n_bands: energy} where energy is None if any momentum-sector
    Hilbert-space dimension exceeds *dim_cutoff*.
    """
    ed_energies: dict[int, float | None] = {}
    for n_bands in (1, 2, 3):
        selected_bands = tuple(range(n_bands))
        print(f"\nBloch-ED: building {n_bands}-band projected Hamiltonian ...")
        proj_ham = build_mote2_three_orbital_projected_hamiltonian(
            selected_bands=selected_bands,
            supercell_matrix=supercell_matrix,
            delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
            t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
            a_m=a_m, ph_conj=ph_conj, V1=V1,
        )
        n_orbitals = int(proj_ham.one_body.shape[0])
        lattice    = proj_ham.lattice
        n_sectors  = lattice.Lx * lattice.Ly

        # Quick upper-bound: if C(n_orbitals, n_fermions) is huge, skip early.
        total_dim = math.comb(n_orbitals, n_fermions)
        if total_dim > dim_cutoff * n_sectors:
            print(
                f"  Skipped: total C({n_orbitals},{n_fermions})={total_dim} "
                f"implies some sector > {dim_cutoff}."
            )
            ed_energies[n_bands] = None
            continue

        # Build all sector bases (cached internally) and find max sector dim.
        max_dim = 0
        for kx in range(lattice.Lx):
            for ky in range(lattice.Ly):
                basis = build_fock_basis(
                    n_orbitals=n_orbitals,
                    n_particles=n_fermions,
                    orbital_momenta=proj_ham.orbital_momenta,
                    lattice_shape=(lattice.Lx, lattice.Ly),
                    momentum_sector=(kx, ky),
                )
                max_dim = max(max_dim, len(basis.states))

        if max_dim > dim_cutoff:
            print(
                f"  Skipped: max sector dim={max_dim} > {dim_cutoff} "
                f"(n_orbitals={n_orbitals})."
            )
            ed_energies[n_bands] = None
            continue

        print(f"  Solving: n_orbitals={n_orbitals}, max sector dim={max_dim} ...")
        sector_results = solve_projected_all_momentum_sectors(
            projected_hamiltonian=proj_ham,
            n_particles=n_fermions,
        )
        all_eigs = [
            res["eigenvalues"]
            for res in sector_results.values()
            if len(res["eigenvalues"]) > 0
        ]
        if not all_eigs:
            ed_energies[n_bands] = None
            continue

        e_gs = float(min(float(ev[0]) for ev in all_eigs))
        ed_energies[n_bands] = e_gs
        print(f"  E_gs({n_bands} Band) = {e_gs:.10f}")

    return ed_energies


def _add_ed_lines(ax, ed_energies: dict[int, float | None]) -> None:
    """Draw horizontal ED reference lines on *ax* for each available band count."""
    for n_bands in (1, 2, 3):
        e = ed_energies.get(n_bands)
        if e is not None:
            ax.axhline(
                e,
                color=_ED_COLORS.get(n_bands, "gray"),
                ls=_ED_LS.get(n_bands, "--"),
                lw=1.5,
                label=f"{n_bands} Band",
                zorder=2,
            )


# ---------------------------------------------------------------------------
# Structure factor
# ---------------------------------------------------------------------------

def _structure_factor(
    samples: np.ndarray,
    positions: np.ndarray,
    q_vectors: np.ndarray,
    n_fermions: int,
) -> np.ndarray:
    phases = np.exp(-1j * (positions @ q_vectors.T))
    rho_q = samples @ phases
    return np.mean(np.abs(rho_q) ** 2, axis=0) / n_fermions


# ---------------------------------------------------------------------------
# History (enables resuming interrupted runs)
# ---------------------------------------------------------------------------

_HIST_DTYPES: dict[str, np.dtype] = {
    "iters":              np.int64,
    "energy_mean":        np.complex128,
    "energy_sigma":       np.float64,
    "energy_variance":    np.float64,
    "energy_std_local":   np.float64,
    "energy_tau":         np.float64,
    "energy_rhat":        np.float64,
    "update_norm_iters":  np.int64,
    "update_norm_values": np.float64,
}


def _load_history(path: Path) -> dict:
    hist = {k: np.array([], dtype=v) for k, v in _HIST_DTYPES.items()}
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        for k, dtype in _HIST_DTYPES.items():
            if k in loaded:
                hist[k] = np.asarray(loaded[k], dtype=dtype)
    return hist


def _concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not a.size:
        return b
    if not b.size:
        return a
    return np.concatenate((a, b))


def _truncate_history(history: dict, max_completed_step: int) -> dict:
    # Written with Codex 03-02-26.
    if max_completed_step <= 0 or history["iters"].size == 0:
        return history
    energy_keep = history["iters"] < max_completed_step
    update_keep = history["update_norm_iters"] < max_completed_step
    if bool(np.all(energy_keep)) and bool(np.all(update_keep)):
        return history
    trimmed = {}
    for key, dtype in _HIST_DTYPES.items():
        values = history[key]
        keep = update_keep if key.startswith("update_norm") else energy_keep
        trimmed[key] = np.asarray(values[keep], dtype=dtype)
    return trimmed


def _checkpoint_step(path: Path) -> int | None:
    # Written with Codex 03-02-26.
    stem = path.stem
    if not stem.startswith("step_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def _select_resume_checkpoint(
    checkpoint_path: Path,
    ckpt_dir: Path,
) -> tuple[Path | None, int | None]:
    # Written with Codex 03-02-26.
    if checkpoint_path.exists():
        return checkpoint_path, None
    best_path = None
    best_step = None
    for candidate in ckpt_dir.glob("step_*.mpack"):
        step = _checkpoint_step(candidate)
        if step is None:
            continue
        if best_step is None or step > best_step:
            best_path = candidate
            best_step = step
    return best_path, best_step


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path, dpi: int = 180) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Job-folder resolution
# ---------------------------------------------------------------------------

def _resolve_job_dir(script_dir: Path, job_name: str) -> tuple[Path, bool]:
    # Written with Codex 03-05-26.
    """Return (job_dir, is_new_job).

    If `raw_data/` already exists alongside the script (i.e. the script is
    running from inside a job folder), treat the script's directory as the
    job folder and resume.  Otherwise create a fresh job folder.
    """
    if (script_dir / "raw_data").exists():
        return script_dir, False
    launch_root = _LAUNCH_ROOT if script_dir.name == "scripts" else script_dir
    return launch_root / "jobs" / job_name, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- GPU check (before job folder exists) ---
    backend = jax.default_backend()
    print("devices:", jax.devices())
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend!r}. "
            "Set CUDA_VISIBLE_DEVICES appropriately."
        )

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    # Physical system — can be overridden via env vars for testing
    _scm_str = os.environ.get("SUPERCELL_MATRIX", "")
    if _scm_str:
        # Parse "a,b,c,d" → [[a,b],[c,d]]
        _vals = [int(x) for x in _scm_str.split(",")]
        supercell_matrix = np.array([[_vals[0], _vals[1]], [_vals[2], _vals[3]]], dtype=np.int64)
    else:
        supercell_matrix = np.array([[3, 0], [0, 3]], dtype=np.int64)

    n_fermions = _env_int("N_FERMIONS", 3)
    V1         = _env_float("V1", 0.3)

    # MoTe2 ideal-band parameters
    delta   = 5.426542963374432
    ez      = 0.0
    t_th1   = 1.0
    t_hh1   = 0.4689334070165771
    t_th2   = -0.07151471515876148
    t_hh3   = 0.037340187144172164
    t_tt1   = -0.0377535889269079
    a_m     = 1.0
    ph_conj = True

    # Optimisation
    sample_type         = _env_str("SAMPLE_TYPE", "FullSum")
    n_iter              = _env_int("N_ITER", 2_000)
    n_samples           = _env_int("N_SAMPLES", 1024 * 4)
    n_discard_per_chain = _env_int("N_DISCARD_PER_CHAIN", 4)
    sweep_size          = _env_int("SWEEP_SIZE", 2 * n_fermions)
    n_chains            = _env_int("N_CHAINS", 1024)
    learning_rate       = _env_float("LEARNING_RATE", 0.05)
    diag_shift          = _env_float("DIAG_SHIFT", 0.001)

    # Post-training observables
    obs_n_samples           = _env_int("OBS_N_SAMPLES", 1_000_000)
    obs_n_discard_per_chain = _env_int("OBS_N_DISCARD_PER_CHAIN", max(64, n_discard_per_chain))

    # MCState chunk size: limits the number of samples per JIT call for both
    # the SR Jacobian and local_estimators.
    # The ViT's O(n_sites²) attention means the full batch
    #   n_samples × max_conn_size × n_heads × n_sites × n_sites
    # is enormous for 81-site systems (>50 GiB with n_samples=4096).
    # chunk_size=32 keeps each kernel at ~32 MB, well within a 24 GB GPU.
    # Use 0 to disable chunking (only for small validation systems).
    _chunk_size_raw = _env_int("CHUNK_SIZE", 32)
    chunk_size: int | None = _chunk_size_raw if _chunk_size_raw > 0 else None

    # ViT architecture
    num_layers        = _env_int("NUM_LAYERS", 2)
    d_model           = _env_int("D_MODEL", 16)
    n_heads           = _env_int("N_HEADS", 2)
    mlp_hidden_factor = _env_int("MLP_HIDDEN_FACTOR", 2)
    output_hidden_dim = _env_int("OUTPUT_HIDDEN_DIM", 16)
    displacement_only_attention = _env_bool("DISPLACEMENT_ONLY_ATTENTION", True)
    slater_init_mode  = _env_str("SLATER_INIT_MODE", "noninteracting")
    
    # -----------------------------------------------------------------------
    # Resolve job folder
    # -----------------------------------------------------------------------

    script_dir = Path(__file__).resolve().parent
    n_sc_cells = abs(int(np.linalg.det(supercell_matrix).round()))
    a, b_  = int(supercell_matrix[0, 0]), int(supercell_matrix[0, 1])
    c, d_  = int(supercell_matrix[1, 0]), int(supercell_matrix[1, 1])
    c_str  = f"m{abs(c)}" if c < 0 else str(c)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = (
        f"vit_scm_{a}_{b_}_{c_str}_{d_}_nf{n_fermions}"
        f"_V1_{V1:.2f}_{sample_type.lower()}_{timestamp}"
    )

    job_dir, is_new_job = _resolve_job_dir(script_dir, job_name)
    raw_data_dir  = job_dir / "raw_data"
    plots_opt_dir = job_dir / "plots_optimization"
    plots_obs_dir = job_dir / "plots_observables"
    ckpt_dir      = job_dir / "checkpoints"
    for folder in (job_dir, raw_data_dir, plots_opt_dir, plots_obs_dir, ckpt_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if is_new_job:
        shutil.copy2(__file__, job_dir / "run_script.py")

    # Redirect stdout/stderr to job_dir/run.log (tee: console + file).
    log_file = open(job_dir / "run.log", "a")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self._streams:
                s.flush()

        def fileno(self):
            return self._streams[0].fileno()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"\n{'=' * 60}")
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Job dir:     {job_dir}")
    print(f"{'=' * 60}\n")

    history_path    = raw_data_dir / "optimization_history.npz"
    checkpoint_path = job_dir / "vstate_variables.mpack"
    resume_checkpoint_path, checkpoint_resume_step = _select_resume_checkpoint(
        checkpoint_path,
        ckpt_dir,
    )

    # -----------------------------------------------------------------------
    # Hamiltonian and geometry
    # -----------------------------------------------------------------------

    graph, hi, ham = build_mote2_three_orbital_hamiltonian(
        supercell_matrix=supercell_matrix,
        n_fermions=n_fermions,
        delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
        t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
        a_m=a_m, ph_conj=ph_conj, V1=V1,
    )

    assert n_fermions <= graph.n_nodes
    assert int(graph.n_nodes) == 3 * n_sc_cells, (
        f"Unexpected site count: n_sites={graph.n_nodes}, expected={3 * n_sc_cells}"
    )
    print(f"n_sites={graph.n_nodes}, n_fermions={hi.n_fermions}")
    print(f"supercell_matrix={supercell_matrix.tolist()}, n_sc_cells={n_sc_cells}")
    print(f"V1={V1}, max_conn_size={ham.max_conn_size}")

    positions, sc_t1, sc_t2, lattice = _build_geometry(supercell_matrix, a_m)
    sc_basis = np.stack([sc_t1, sc_t2], axis=0)  # (2, 2) supercell basis, rows = vectors

    # Non-interacting reference energy (printed for V1=0 validation)
    e_ni = _noninteracting_energy(
        supercell_matrix, n_fermions,
        delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
        t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
        a_m=a_m, ph_conj=ph_conj,
    )
    print(f"Non-interacting reference energy (V1=0): {e_ni:.10f}")

    # -----------------------------------------------------------------------
    # Bloch-ED reference energies (1-band, 2-band, 3-band projections)
    # -----------------------------------------------------------------------

    print("\nComputing Bloch-ED reference energies ...")
    ed_energies = _compute_mote2_ed_reference_energies(
        supercell_matrix=supercell_matrix,
        n_fermions=n_fermions,
        delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
        t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
        a_m=a_m, ph_conj=ph_conj, V1=V1,
    )
    print()

    # -----------------------------------------------------------------------
    # Pair data for the ViT
    # -----------------------------------------------------------------------

    pair_classes_hash, pair_distances_hash, pair_classes_np, pair_distances_np = (
        _make_pair_data(positions, sc_t1, sc_t2)
    )
    n_pair_classes = len(pair_distances_hash)
    print(f"Number of distinct pair-distance classes: {n_pair_classes}")
    print(f"displacement_only_attention={displacement_only_attention}")

    # -----------------------------------------------------------------------
    # Optional non-interacting Slater initialisation
    # -----------------------------------------------------------------------

    slater_initial_m_orbitals = None
    if slater_init_mode == "noninteracting":
        slater_initial_m_orbitals = noninteracting_slater_orbitals_mote2_three_orbital(
            supercell_matrix=supercell_matrix,
            n_fermions=n_fermions,
            delta=delta, ez=ez, t_th1=t_th1, t_hh1=t_hh1,
            t_th2=t_th2, t_hh3=t_hh3, t_tt1=t_tt1,
            a_m=a_m, ph_conj=ph_conj,
        )
        print("Initializing Slater determinant from non-interacting orbitals.")
    elif slater_init_mode != "random":
        raise ValueError(f"Unknown SLATER_INIT_MODE={slater_init_mode!r}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = LogSlaterSpatialViT(
        hilbert=hi,
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        pair_classes=pair_classes_hash,
        pair_distances=pair_distances_hash,
        displacement_only_attention=displacement_only_attention,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=slater_initial_m_orbitals,
        vit_param_dtype=jnp.float64,
        mlp_hidden_factor=mlp_hidden_factor,
        output_hidden_dim=output_hidden_dim,
        xi_epsilon=1.0e-6,
    )

    # -----------------------------------------------------------------------
    # Variational state
    # -----------------------------------------------------------------------

    sampler = nk.sampler.MetropolisFermionHop(
        hi, graph=graph, n_chains=n_chains, sweep_size=sweep_size,
    )

    if sample_type == "MC":
        vstate = nk.vqs.MCState(
            sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size,
        )
        ham_sr = ham
    elif sample_type == "FullSum":
        ham_sr = ham.to_fermionoperator2nd()
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        raise ValueError(f"Unknown sample_type={sample_type!r}")

    print(f"total # of wavefunction parameters: {nk.jax.tree_size(vstate.parameters)}")

    # -----------------------------------------------------------------------
    # Load checkpoint (resume support)
    # -----------------------------------------------------------------------

    history_prev = _load_history(history_path)
    history_resume_step = int(history_prev["iters"][-1]) + 1 if history_prev["iters"].size > 0 else 0
    if checkpoint_resume_step is None:
        start_step = history_resume_step
    else:
        start_step = checkpoint_resume_step
        if history_resume_step != start_step:
            relation = "ahead of" if history_resume_step > start_step else "behind"
            print(
                "History extends "
                f"{relation} checkpoint state; truncating/using checkpoint step {start_step}."
            )
        history_prev = _truncate_history(history_prev, start_step)

    if start_step > 0:
        print(f"Resuming from step {start_step} ({history_prev['iters'].size} steps in history).")
    else:
        print("Starting fresh from step 0.")

    if resume_checkpoint_path is not None:
        vstate.variables = flax.serialization.from_bytes(
            vstate.variables, resume_checkpoint_path.read_bytes()
        )
        print(f"Loaded checkpoint: {resume_checkpoint_path}")
    elif start_step > 0:
        raise FileNotFoundError(
            "Expected checkpoint for resumption, but neither "
            f"{checkpoint_path} nor any step_*.mpack file was found."
        )

    # -----------------------------------------------------------------------
    # Optimise — with combined callback: diagnostics + periodic checkpoints
    # -----------------------------------------------------------------------

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham_sr, optimizer, variational_state=vstate, diag_shift=diag_shift, mode="complex",
    )

    # 10 evenly-spaced checkpoints during this run.
    # step is 0-indexed within the run; we save after steps
    # checkpoint_every-1, 2*checkpoint_every-1, ..., (absolute step start_step+k).
    _base_callback   = make_optimization_callback(job_dir)
    _checkpoint_every = max(1, n_iter // 10)
    # Steps (0-indexed) at which to save: checkpoint_every-1, 2*checkpoint_every-1, …
    _checkpoint_steps = set(
        range(_checkpoint_every - 1, n_iter, _checkpoint_every)
    )

    def _callback(step: int, log_data: dict, driver) -> bool:
        _base_callback(step, log_data, driver)
        if step in _checkpoint_steps:
            abs_step = start_step + step + 1
            ckpt = ckpt_dir / f"step_{abs_step}.mpack"
            ckpt.write_bytes(flax.serialization.to_bytes(vstate.variables))
            print(f"[Checkpoint] Saved at absolute step {abs_step}: {ckpt}", flush=True)
        return True

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=_callback)

    # -----------------------------------------------------------------------
    # Save final checkpoint and parameters
    # -----------------------------------------------------------------------

    checkpoint_path.write_bytes(flax.serialization.to_bytes(vstate.variables))
    param_leaves, param_treedef = jax.tree_util.tree_flatten(vstate.parameters)
    np.savez(
        job_dir / "vstate_parameters_leaves.npz",
        *[np.asarray(x) for x in param_leaves],
    )
    (job_dir / "vstate_parameters_treedef.txt").write_text(repr(param_treedef) + "\n")

    end_step = start_step + n_iter
    log.serialize(job_dir / f"runtime_log_step{start_step}_to_step{end_step}")
    log.serialize(job_dir / "runtime_log")

    # -----------------------------------------------------------------------
    # Collect and save history
    # -----------------------------------------------------------------------

    energy_hist = log.data.get("Energy", None)
    if energy_hist is None:
        iters_new = mean_new = sigma_new = var_new = tau_new = rhat_new = np.array([], dtype=np.float64)
        iters_new = np.array([], dtype=np.int64)
        mean_new  = np.array([], dtype=np.complex128)
        sigma_new = np.array([], dtype=np.float64)
        var_new   = np.array([], dtype=np.float64)
        tau_new   = np.array([], dtype=np.float64)
        rhat_new  = np.array([], dtype=np.float64)
    else:
        iters_new = np.asarray(energy_hist.iters, dtype=np.int64) + start_step
        mean_new  = np.asarray(energy_hist["Mean"],     dtype=np.complex128)
        sigma_new = np.asarray(energy_hist["Sigma"],    dtype=np.float64)
        var_new   = np.asarray(energy_hist["Variance"], dtype=np.float64)
        try:
            tau_new = np.asarray(energy_hist["TauCorr"], dtype=np.float64)
        except Exception:
            tau_new = np.array([], dtype=np.float64)
        try:
            rhat_new = np.asarray(energy_hist["R_hat"], dtype=np.float64)
        except Exception:
            rhat_new = np.array([], dtype=np.float64)

    un_hist = log.data.get("UpdateNormL2", None)
    if un_hist is not None:
        un_iters_new  = np.asarray(un_hist.iters, dtype=np.int64) + start_step
        un_values_new = np.asarray(un_hist, dtype=np.float64)
    else:
        un_iters_new  = np.array([], dtype=np.int64)
        un_values_new = np.array([], dtype=np.float64)

    hist = {
        "iters":              _concat(history_prev["iters"],              iters_new),
        "energy_mean":        _concat(history_prev["energy_mean"],        mean_new),
        "energy_sigma":       _concat(history_prev["energy_sigma"],       sigma_new),
        "energy_variance":    _concat(history_prev["energy_variance"],    var_new),
        "energy_tau":         _concat(history_prev["energy_tau"],         tau_new),
        "energy_rhat":        _concat(history_prev["energy_rhat"],        rhat_new),
        "update_norm_iters":  _concat(history_prev["update_norm_iters"],  un_iters_new),
        "update_norm_values": _concat(history_prev["update_norm_values"], un_values_new),
    }
    hist["energy_std_local"] = np.sqrt(np.maximum(np.real(hist["energy_variance"]), 0.0))
    np.savez(history_path, **hist)

    iters            = hist["iters"]
    energy_mean      = hist["energy_mean"]
    energy_sigma     = hist["energy_sigma"]
    energy_std_local = hist["energy_std_local"]
    energy_tau       = hist["energy_tau"]
    energy_rhat      = hist["energy_rhat"]
    un_iters         = hist["update_norm_iters"]
    un_values        = hist["update_norm_values"]
    total_steps      = int(iters[-1]) + 1 if iters.size > 0 else start_step

    # -----------------------------------------------------------------------
    # Optimisation plots
    # -----------------------------------------------------------------------

    def _energy_band(ax, x, mean, sigma, **kw):
        ax.plot(x, np.real(mean), **kw)
        ax.fill_between(x, np.real(mean) - sigma, np.real(mean) + sigma,
                        color=kw.get("color", "tab:blue"), alpha=0.2, linewidth=0)

    # Full energy trace
    fig, ax = plt.subplots(figsize=(7, 4))
    _energy_band(ax, iters, energy_mean, energy_sigma, lw=1.8, color="tab:blue", label="VMC_SR")
    if V1 == 0.0:
        ax.axhline(e_ni, color="tab:purple", ls="--", lw=1.5, label=f"NI ref ({e_ni:.6f})")
    _add_ed_lines(ax, ed_energies)
    ax.set(title="MoTe2 three-orbital: ViT × Slater optimisation",
           xlabel="Optimisation step", ylabel="Energy")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_fig(fig, plots_opt_dir / "energy_vs_step.png")

    # Final 80%
    i0 = max(int(math.floor(0.2 * iters.size)), 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    if iters.size > 0:
        _energy_band(ax, iters[i0:], energy_mean[i0:], energy_sigma[i0:],
                     lw=1.8, color="tab:blue", label="VMC_SR")
        if V1 == 0.0:
            ax.axhline(e_ni, color="tab:purple", ls="--", lw=1.5, label=f"NI ref ({e_ni:.6f})")
        _add_ed_lines(ax, ed_energies)
        ax.legend()
    ax.set(xlabel="Optimisation step", ylabel="Energy")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "energy_vs_step_final80pct.png")

    # Local-energy std dev
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.log10(np.maximum(energy_std_local, 1e-30)), color="tab:purple", lw=1.5)
    ax.set(title="Local energy standard deviation",
           xlabel="Optimisation step", ylabel="log10 Std(E_loc)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "local_energy_std_log10.png")

    # Update norm
    fig, ax = plt.subplots(figsize=(7, 4))
    if un_values.size > 0:
        ax.plot(un_iters, un_values, color="tab:red", lw=1.5)
        ax.set_yscale("log")
    ax.set(title="Update norm", xlabel="Optimisation step",
           ylabel=r"||$\Delta\theta$||$_2$")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "update_norm.png")

    # Sampling diagnostics
    fig, ax = plt.subplots(figsize=(7, 4))
    if energy_tau.size > 0:
        nt = min(iters.size, energy_tau.size)
        ax.plot(iters[:nt], energy_tau[:nt], color="tab:orange", lw=1.5, label="TauCorr")
        if energy_rhat.size > 0:
            nr = min(iters.size, energy_rhat.size)
            ax.plot(iters[:nr], energy_rhat[:nr], color="tab:brown", lw=1.5, label="R_hat")
        ax.legend()
    ax.set(title="Sampling diagnostics", xlabel="Optimisation step")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "sampling_diagnostics.png")

    # Save a copy of the run script in plots_opt_dir for reproducibility
    shutil.copy2(job_dir / "run_script.py", plots_opt_dir / "run_script_used.py")

    # -----------------------------------------------------------------------
    # Post-training observables
    # -----------------------------------------------------------------------

    # FullSumState has no .sample() method; for observable collection we always
    # use an MCState (either the existing one, or a temporary one for FullSum runs).
    if isinstance(vstate, nk.vqs.FullSumState):
        # For FullSum validation runs, cap obs_n_samples to keep it fast.
        _obs_n_samples  = min(obs_n_samples, 100_000)
        _obs_n_discard  = obs_n_discard_per_chain
        _obs_sampler = nk.sampler.MetropolisFermionHop(
            hi, graph=graph, n_chains=min(n_chains, 256), sweep_size=sweep_size,
        )
        _obs_vstate = nk.vqs.MCState(
            _obs_sampler, model,
            n_samples=_obs_n_samples,
            n_discard_per_chain=_obs_n_discard,
        )
        _obs_vstate.variables = vstate.variables
        print(
            f"\nSampling observables (FullSumState→MCState) at step {total_steps}: "
            f"n_samples={_obs_n_samples}, n_discard={_obs_n_discard}"
        )
        samples_obs = (
            np.asarray(_obs_vstate.sample())
            .reshape(-1, hi.size)
            .astype(np.float64)
        )
    else:
        print(
            f"\nSampling observables at step {total_steps}: "
            f"n_samples={obs_n_samples}, n_discard={obs_n_discard_per_chain}"
        )
        samples_obs = (
            np.asarray(
                vstate.sample(n_samples=obs_n_samples, n_discard_per_chain=obs_n_discard_per_chain)
            )
            .reshape(-1, hi.size)
            .astype(np.float64)
        )

    # Pair correlation: minimum-image distance is computed inside pair_correlation_cartesian
    # using the supercell basis vectors (sc_basis rows = sc_t1, sc_t2) with Lx=Ly=1
    # and pbc=[True, True], so each pairwise distance is the shortest image distance.
    pc = pair_correlation_cartesian(
        samples=samples_obs,
        positions=positions,
        basis_vectors=sc_basis,   # rows are sc_t1, sc_t2
        pbc=np.array([True, True]),
        Lx=1,
        Ly=1,
        n_fermions=n_fermions,
    )
    charge_density = pc["charge_density"]
    corr_matrix    = pc["corr_matrix"]
    r_values_plot  = pc["r_values_plot"]
    g_r_plot       = pc["g_r_plot"]

    # Map pair correlation to 2D using minimum-image positions
    rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(
        positions=positions,
        basis_coords=np.asarray(lattice.site_to_cell),
        translations=pc["translations"],
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
    )

    # Structure factor: q-vectors folded to WS-BZ of primitive MoTe2 lattice
    q_list_raw, q_list = _supercell_q_vectors(lattice, a_m)
    s_q = _structure_factor(samples_obs, positions, q_list, n_fermions)
    q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot = radial_average_structure_factor(
        q_list, s_q, set_q0_to_zero=True,
    )

    # Fold positions into supercell rhombus for spatial plots
    positions_folded = _fold_positions(positions, sc_t1, sc_t2)

    # -----------------------------------------------------------------------
    # Save raw observables
    # -----------------------------------------------------------------------

    step_suffix  = f"_step{total_steps}"
    obs_raw_path = raw_data_dir / f"observables_data{step_suffix}.npz"
    np.savez(
        obs_raw_path,
        samples_obs=samples_obs.astype(np.int8),
        charge_density=charge_density,
        positions=positions,
        positions_folded=positions_folded,
        corr_matrix=corr_matrix,
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
        q_list_raw=q_list_raw,
        q_list=q_list,
        s_q=s_q,
        q_shell_unique=q_shell_unique,
        s_q_abs=s_q_abs,
        s_q_abs_err=s_q_abs_err,
        q_abs=q_abs,
        s_q_plot=s_q_plot,
        rel_min=rel_min,
        g_site=g_site,
        r_site=r_site,
        origin_idx=np.array([origin_idx], dtype=np.int64),
        obs_n_samples=np.array([samples_obs.shape[0]], dtype=np.int64),
        obs_n_discard_per_chain=np.array([obs_n_discard_per_chain], dtype=np.int64),
        optimization_step=np.array([total_steps], dtype=np.int64),
        sc_basis=sc_basis,
        sc_t1=sc_t1,
        sc_t2=sc_t2,
        e_ni=np.array([e_ni]),
        pair_classes=pair_classes_np,
        pair_distances=pair_distances_np,
    )

    # -----------------------------------------------------------------------
    # Observable plots
    # -----------------------------------------------------------------------

    rhombus = np.array([np.zeros(2), sc_t1, sc_t1 + sc_t2, sc_t2, np.zeros(2)])
    rhombus_centered = np.array([
        -0.5 * sc_t1 - 0.5 * sc_t2,
         0.5 * sc_t1 - 0.5 * sc_t2,
         0.5 * sc_t1 + 0.5 * sc_t2,
        -0.5 * sc_t1 + 0.5 * sc_t2,
        -0.5 * sc_t1 - 0.5 * sc_t2,
    ])

    # Charge density
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sc_ = ax.scatter(positions_folded[:, 0], positions_folded[:, 1],
                     c=charge_density, s=140, cmap="viridis",
                     vmin=0.0, vmax=float(np.max(charge_density)))
    ax.plot(rhombus[:, 0], rhombus[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title=r"Charge density $\langle n_i \rangle$",
           xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(sc_, ax=ax, label=r"$\langle n_i \rangle$")
    _save_fig(fig, plots_obs_dir / f"charge_density{step_suffix}.png")

    # Pair correlation — radial (distances are minimum-image distances)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax.set(title="Pair correlation (minimum-image distances)",
           xlabel="Distance r (minimum image)", ylabel="g(r)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"pair_correlation_radial{step_suffix}.png")

    # Pair correlation — 2D minimum image map
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc2 = ax.scatter(rel_min[:, 0], rel_min[:, 1], c=g_site, s=180, cmap="viridis")
    ax.scatter([0.0], [0.0], s=220, facecolors="none", edgecolors="black", linewidths=1.1)
    ax.plot(rhombus_centered[:, 0], rhombus_centered[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title="Pair correlation mapped to 2D (minimum image)",
           xlabel="x (minimum image)", ylabel="y (minimum image)", aspect="equal")
    fig.colorbar(sc2, ax=ax, label="g(r)")
    _save_fig(fig, plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png")

    # Structure factor — radial (q-values are WS-BZ representatives)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(q_shell_unique, s_q_abs, yerr=s_q_abs_err,
                fmt="o-", ms=4, lw=1.2, capsize=2, color="tab:blue")
    ax.set(title="Static structure factor (radial, WS-BZ folded)",
           xlabel=r"$|\mathbf{q}|$ (WS-BZ)", ylabel="S(q)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"static_structure_factor{step_suffix}.png")

    # Structure factor — q-space map (q-vectors folded to WS-BZ)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc3 = ax.scatter(q_list[:, 0], q_list[:, 1], c=s_q_plot.real,
                     s=120, cmap="magma", edgecolors="black", linewidths=0.25)
    ax.set(title="Static structure factor in q-space (WS-BZ folded)",
           xlabel=r"$q_x$", ylabel=r"$q_y$", aspect="equal")
    ax.grid(alpha=0.2)
    fig.colorbar(sc3, ax=ax, label=r"$S(\mathbf{q})$")
    _save_fig(fig, plots_obs_dir / f"static_structure_factor_qspace{step_suffix}.png")

    # Save run script to observable plots dir for reference
    shutil.copy2(job_dir / "run_script.py", plots_obs_dir / f"run_script_used{step_suffix}.py")

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------

    summary = {
        "job_dir":    str(job_dir),
        "backend":    backend,
        "model_type": "vit_slater",
        "sample_type": sample_type,
        "system": {
            "supercell_matrix": supercell_matrix.tolist(),
            "n_supercell_cells": n_sc_cells,
            "n_sites": int(graph.n_nodes),
            "n_fermions": n_fermions,
            "V1": V1, "delta": delta, "ez": ez,
            "t_th1": t_th1, "t_hh1": t_hh1, "t_th2": t_th2,
            "t_hh3": t_hh3, "t_tt1": t_tt1, "a_m": a_m, "ph_conj": ph_conj,
            "e_ni": e_ni,
        },
        "network": {
            "num_layers": num_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "mlp_hidden_factor": mlp_hidden_factor,
            "output_hidden_dim": output_hidden_dim,
            "displacement_only_attention": displacement_only_attention,
            "slater_init_mode": slater_init_mode,
            "n_pair_classes": n_pair_classes,
        },
        "optimization": {
            "n_iter_this_run": n_iter,
            "start_step": start_step,
            "end_step_exclusive": end_step,
            "total_steps": total_steps,
            "n_samples": n_samples,
            "n_discard_per_chain": n_discard_per_chain,
            "n_chains": n_chains,
            "sweep_size": sweep_size,
            "optimizer": f"Sgd(learning_rate={learning_rate})",
            "diag_shift": diag_shift,
            "mode": "complex",
            "checkpoint_every": _checkpoint_every,
            "chunk_size": chunk_size,
        },
        "observables": {
            "obs_n_samples": int(samples_obs.shape[0]),
            "obs_n_discard_per_chain": obs_n_discard_per_chain,
            "optimization_step": total_steps,
        },
        "n_wavefunction_params": int(nk.jax.tree_size(vstate.parameters)),
        "final_energy_real":  float(np.real(energy_mean[-1]))  if energy_mean.size  > 0 else None,
        "final_energy_sigma": float(energy_sigma[-1])           if energy_sigma.size > 0 else None,
        "final_update_norm":  float(un_values[-1])              if un_values.size    > 0 else None,
        "e_ni": e_ni,
        "error_vs_ni": (
            float(np.real(energy_mean[-1])) - e_ni if energy_mean.size > 0 else None
        ),
        "ed_reference_energies": {
            str(n_bands): (None if e is None else float(e))
            for n_bands, e in ed_energies.items()
        },
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nNon-interacting ref energy = {e_ni:.10f}")
    print(f"Final energy (Re)          = {np.real(energy_mean[-1]):.10f}")
    if energy_mean.size > 0:
        print(f"Error vs NI ref            = {float(np.real(energy_mean[-1])) - e_ni:.6e}")
    if un_values.size > 0:
        print(f"Final update norm          = {float(un_values[-1]):.6e}")
    print(f"Final std(E_loc)           = {float(energy_std_local[-1]):.10f}")
    for n_bands, e in sorted(ed_energies.items()):
        tag = f"{e:.10f}" if e is not None else "skipped"
        print(f"ED ref ({n_bands} Band)            = {tag}")
    print(f"Observables raw:    {obs_raw_path}")
    print(f"Optimisation plots: {plots_opt_dir}")
    print(f"Observable plots:   {plots_obs_dir}")


if __name__ == "__main__":
    main()
