"""ViT x Slater parent run script for the Haldane model.

Running this script creates a timestamped job folder under `jobs/`, copies
itself there as `run_script.py`, and executes VMC optimisation followed by
observable collection. Re-running `run_script.py` from inside its own job
folder resumes an interrupted run (checkpoint loading is automatic).

Directory layout after one run
------------------------------
results/haldane/vit/
├── run.py                          <- this file (never modified by a run)
└── jobs/
    └── vit_scm_3_0_0_3_nf3_..._YYYYMMDD_HHMMSS/
        ├── run_script.py           <- exact copy of run.py at launch time
        ├── run.log                 <- stdout + stderr
        ├── gpu_mem.log             <- per-step GPU memory
        ├── summary.json
        ├── vstate_variables.mpack  <- final checkpoint (full variables)
        ├── vstate_parameters_leaves.npz
        ├── vstate_parameters_treedef.txt
        ├── runtime_log.json
        ├── checkpoints/
        ├── raw_data/
        ├── plots_optimization/
        └── plots_observables/

Environment variables (all optional)
------------------------------------
  CUDA_VISIBLE_DEVICES        GPU index (default: 0)
  SUPERCELL_MATRIX            2x2 integer matrix as "a,b,c,d" (default: "3,0,0,3")
                              Only diagonal rectangular supercells are currently
                              supported: off-diagonal entries must be zero.
  N_FERMIONS                  Number of fermions (default: 3)
  V1                          Interaction strength (default: 10.0)
  SAMPLE_TYPE                 MC | FullSum (default: MC)
  N_ITER                      Optimisation steps (default: 5000)
  N_SAMPLES                   MC samples per step (default: 4096)
  N_DISCARD_PER_CHAIN         Discard per chain per step (default: 4)
  SWEEP_SIZE                  MCMC sweep size (default: n_fermions)
  N_CHAINS                    Number of MCMC chains (default: 1024)
  LEARNING_RATE               SGD learning rate (default: 0.05)
  DIAG_SHIFT                  SR diagonal shift (default: 0.01)
  OBS_N_SAMPLES               Samples for post-training observables (default: 1_000_000)
  OBS_N_DISCARD_PER_CHAIN     Discard for observable sampling (default: max(64, N_DISCARD))
  CHUNK_SIZE                  MCState chunk size for grad + local_estimators (default: 0)
                              Set to 0 to disable chunking.
  NUM_LAYERS                  ViT encoder depth (default: 2)
  D_MODEL                     Token embedding dimension (default: 16)
  N_HEADS                     Attention heads (default: 2)
  MLP_HIDDEN_FACTOR           MLP hidden-layer factor (default: 2)
  OUTPUT_HIDDEN_DIM           ViT output hidden dimension (default: 16)
  DISPLACEMENT_ONLY_ATTENTION 1 | 0. If 1, alpha_ij depends only on relative
                              displacement class. If 0, alpha_ij is independent
                              for every ordered pair (i, j). (default: 1)
  SLATER_INIT_MODE            random | noninteracting (default: noninteracting)
  T1                          NN hopping (default: 1.0)
  T2                          NNN hopping (default: -1/(4*cos(phi)))
  PHI                         Haldane phase (default: 0.65)
  M                           Sublattice mass (default: 0.0)
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

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_autotune_level" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_autotune_level=0").lstrip()

import flax.serialization
import jax

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
    build_haldane_lattice_embedding,
    build_haldane_projected_hamiltonian,
    build_haldane_real_space_terms,
    solve_projected_all_momentum_sectors,
)
from aidan_custom.geometry import (
    PRIMITIVE_A1,
    PRIMITIVE_A2,
    SUBLATTICE_A_OFFSET,
    SUBLATTICE_B_OFFSET,
    fold_to_shortest_k,
    reciprocal_vectors,
)
from aidan_custom.haldane_model import (
    build_haldane_hamiltonian,
    noninteracting_slater_orbitals_haldane,
)
from aidan_custom.models import (
    LogSlaterSpatialViT,
    make_translation_equivariant_pair_data,
)
from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
)
from aidan_custom.optimization import make_optimization_callback


_ED_COLORS = {1: "tab:orange", 2: "tab:green"}
_ED_LS = {1: "--", 2: "-."}

_HIST_DTYPES: dict[str, np.dtype] = {
    "iters": np.int64,
    "energy_mean": np.complex128,
    "energy_sigma": np.float64,
    "energy_variance": np.float64,
    "energy_std_local": np.float64,
    "energy_tau": np.float64,
    "energy_rhat": np.float64,
    "update_norm_iters": np.int64,
    "update_norm_values": np.float64,
}


def _env_int(key: str, default: int) -> int:
    # Written with Codex 03-02-26.
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    # Written with Codex 03-02-26.
    return float(os.environ.get(key, str(default)))


def _env_str(key: str, default: str) -> str:
    # Written with Codex 03-02-26.
    return os.environ.get(key, default)


def _parse_supercell_matrix(raw: str) -> np.ndarray:
    # Written with Codex 03-02-26.
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != 4:
        raise ValueError(
            "SUPERCELL_MATRIX must have exactly 4 comma-separated integers, "
            f"got {raw!r}."
        )
    matrix = np.asarray(values, dtype=np.int64).reshape(2, 2)
    det = int(round(np.linalg.det(matrix)))
    if det == 0:
        raise ValueError(f"SUPERCELL_MATRIX must be invertible, got {matrix.tolist()}.")
    return matrix


def _diagonal_shape_from_supercell(supercell_matrix: np.ndarray) -> tuple[int, int]:
    # Written with Codex 03-02-26.
    scm = np.asarray(supercell_matrix, dtype=np.int64)
    if scm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 supercell matrix, got {scm.shape}.")
    if int(scm[0, 1]) != 0 or int(scm[1, 0]) != 0:
        raise NotImplementedError(
            "This Haldane launcher currently supports only diagonal rectangular "
            "supercells because `build_haldane_hamiltonian` only accepts Lx and Ly. "
            f"Received SUPERCELL_MATRIX={scm.tolist()}."
        )
    lx = int(scm[0, 0])
    ly = int(scm[1, 1])
    if lx <= 0 or ly <= 0:
        raise ValueError(
            "Diagonal supercell entries must be positive, "
            f"got {scm.tolist()}."
        )
    return lx, ly


def _build_geometry(
    supercell_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    # Written with Codex 03-02-26.
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    lattice = build_haldane_lattice_embedding(Lx=lx, Ly=ly)
    a1 = np.asarray(PRIMITIVE_A1, dtype=np.float64)
    a2 = np.asarray(PRIMITIVE_A2, dtype=np.float64)
    offsets = np.array(
        [
            np.asarray(SUBLATTICE_A_OFFSET, dtype=np.float64),
            np.asarray(SUBLATTICE_B_OFFSET, dtype=np.float64),
        ]
    )

    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell
    positions = np.empty((n_sites, 2), dtype=np.float64)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        bravais = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = (
            float(bravais[0]) * a1
            + float(bravais[1]) * a2
            + offsets[int(orb)]
        )

    sc_t1 = float(lattice.supercell_matrix[0, 0]) * a1 + float(lattice.supercell_matrix[1, 0]) * a2
    sc_t2 = float(lattice.supercell_matrix[0, 1]) * a1 + float(lattice.supercell_matrix[1, 1]) * a2
    return positions, sc_t1, sc_t2, lattice


def _fold_positions(
    positions: np.ndarray,
    sc_t1: np.ndarray,
    sc_t2: np.ndarray,
) -> np.ndarray:
    # Written with Codex 03-02-26.
    sc_cols = np.column_stack([sc_t1, sc_t2])
    frac = np.linalg.solve(sc_cols, positions.T).T
    frac_folded = frac - np.floor(frac)
    return (sc_cols @ frac_folded.T).T


def _supercell_q_vectors(lattice) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 03-02-26.
    b1, b2 = reciprocal_vectors(
        np.asarray(PRIMITIVE_A1, dtype=np.float64),
        np.asarray(PRIMITIVE_A2, dtype=np.float64),
    )
    q_raw = np.array(
        [
            lattice.kpoint_coefficients[kx, ky][0] * b1
            + lattice.kpoint_coefficients[kx, ky][1] * b2
            for kx in range(lattice.Lx)
            for ky in range(lattice.Ly)
        ],
        dtype=np.float64,
    )
    q_folded = np.array(
        [fold_to_shortest_k(q, b1, b2, search_radius=8) for q in q_raw],
        dtype=np.float64,
    )
    return q_raw, q_folded


def _make_pair_data(
    positions: np.ndarray,
    sc_t1: np.ndarray,
    sc_t2: np.ndarray,
) -> tuple[tuple, tuple, np.ndarray, np.ndarray]:
    # Written with Codex 03-02-26.
    sc_basis = np.stack([sc_t1, sc_t2], axis=0)
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


def _noninteracting_energy(
    supercell_matrix: np.ndarray,
    *,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    n_fermions: int,
) -> float:
    # Written with Codex 03-02-26.
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    _, one_body_real_space, _ = build_haldane_real_space_terms(
        Lx=lx,
        Ly=ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=0.0,
    )
    evals = np.linalg.eigvalsh(np.asarray(one_body_real_space, dtype=np.complex128))
    return float(np.real(evals[:n_fermions].sum()))


def _compute_haldane_ed_reference_energies(
    supercell_matrix: np.ndarray,
    n_fermions: int,
    *,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    V1: float,
    dim_cutoff: int = 100_000,
) -> dict[int, float | None]:
    # Written with Codex 03-02-26.
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    ed_energies: dict[int, float | None] = {}
    for n_bands in (1, 2):
        selected_bands = tuple(range(n_bands))
        print(f"\nBloch-ED: building {n_bands}-band projected Hamiltonian ...")
        proj_ham = build_haldane_projected_hamiltonian(
            Lx=lx,
            Ly=ly,
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
            selected_bands=selected_bands,
            V1=V1,
        )
        n_orbitals = int(proj_ham.one_body.shape[0])
        lattice = proj_ham.lattice
        n_sectors = lattice.Lx * lattice.Ly

        total_dim = math.comb(n_orbitals, n_fermions)
        if total_dim > dim_cutoff * n_sectors:
            print(
                f"  Skipped: total C({n_orbitals},{n_fermions})={total_dim} "
                f"implies some sector > {dim_cutoff}."
            )
            ed_energies[n_bands] = None
            continue

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
    # Written with Codex 03-02-26.
    for n_bands in (1, 2):
        e_val = ed_energies.get(n_bands)
        if e_val is None:
            continue
        ax.axhline(
            e_val,
            color=_ED_COLORS[n_bands],
            ls=_ED_LS[n_bands],
            lw=1.5,
            label=f"{n_bands} Band",
            zorder=2,
        )


def _structure_factor(
    samples: np.ndarray,
    positions: np.ndarray,
    q_vectors: np.ndarray,
    n_fermions: int,
) -> np.ndarray:
    # Written with Codex 03-02-26.
    phases = np.exp(-1j * (positions @ q_vectors.T))
    rho_q = samples @ phases
    return np.mean(np.abs(rho_q) ** 2, axis=0) / n_fermions


def _load_history(path: Path) -> dict:
    # Written with Codex 03-02-26.
    hist = {key: np.array([], dtype=dtype) for key, dtype in _HIST_DTYPES.items()}
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        for key, dtype in _HIST_DTYPES.items():
            if key in loaded:
                hist[key] = np.asarray(loaded[key], dtype=dtype)
    return hist


def _concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Written with Codex 03-02-26.
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


def _save_fig(fig, path: Path, dpi: int = 180) -> None:
    # Written with Codex 03-02-26.
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _resolve_job_dir(script_dir: Path, job_name: str) -> tuple[Path, bool]:
    # Written with Codex 03-02-26.
    if (script_dir / "raw_data").exists():
        return script_dir, False
    return script_dir / "jobs" / job_name, True


def main() -> None:
    # Written with Codex 03-02-26.
    backend = jax.default_backend()
    print("devices:", jax.devices())
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend!r}. "
            "Set CUDA_VISIBLE_DEVICES appropriately."
        )

    scm_str = os.environ.get("SUPERCELL_MATRIX", "")
    if scm_str:
        supercell_matrix = _parse_supercell_matrix(scm_str)
    else:
        supercell_matrix = np.array([[4, 0], [0, 3]], dtype=np.int64)

    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    n_fermions = _env_int("N_FERMIONS", 4)
    V1 = _env_float("V1", 1.0)

    t1 = _env_float("T1", 1.0)
    phi = _env_float("PHI", 0.65)
    t2 = _env_float("T2", -1.0 / (4.0 * math.cos(phi)))
    m = _env_float("M", 0.0)

    sample_type = _env_str("SAMPLE_TYPE", "FullSum")
    n_iter = _env_int("N_ITER", 500)
    n_samples = _env_int("N_SAMPLES", 1024 * 4)
    n_discard_per_chain = _env_int("N_DISCARD_PER_CHAIN", 4)
    sweep_size = _env_int("SWEEP_SIZE", n_fermions)
    n_chains = _env_int("N_CHAINS", 1024)
    learning_rate = _env_float("LEARNING_RATE", 0.1)
    diag_shift = _env_float("DIAG_SHIFT", 0.01)

    obs_n_samples = _env_int("OBS_N_SAMPLES", 1_000_000)
    obs_n_discard_per_chain = _env_int(
        "OBS_N_DISCARD_PER_CHAIN",
        max(64, n_discard_per_chain),
    )

    chunk_size_raw = _env_int("CHUNK_SIZE", 0)
    chunk_size: int | None = chunk_size_raw if chunk_size_raw > 0 else None

    # ViT architecture
    num_layers        = _env_int("NUM_LAYERS", 4)
    d_model           = _env_int("D_MODEL", 32)
    n_heads           = _env_int("N_HEADS", 2)
    mlp_hidden_factor = _env_int("MLP_HIDDEN_FACTOR", 2)
    output_hidden_dim = _env_int("OUTPUT_HIDDEN_DIM", 32)
    displacement_only_attention = bool(
        int(os.environ.get("DISPLACEMENT_ONLY_ATTENTION", "0"))
    )
    slater_init_mode  = _env_str("SLATER_INIT_MODE", "noninteracting")

    script_dir = Path(__file__).resolve().parent
    n_sc_cells = abs(int(np.linalg.det(supercell_matrix).round()))
    a, b_ = int(supercell_matrix[0, 0]), int(supercell_matrix[0, 1])
    c, d_ = int(supercell_matrix[1, 0]), int(supercell_matrix[1, 1])
    c_str = f"m{abs(c)}" if c < 0 else str(c)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = (
        f"vit_scm_{a}_{b_}_{c_str}_{d_}_nf{n_fermions}"
        f"_t1_{t1:.2f}_t2_{t2:.2f}_m_{m:.2f}_phi_{phi:.2f}"
        f"_V1_{V1:.2f}_{sample_type.lower()}_{timestamp}"
    )

    job_dir, is_new_job = _resolve_job_dir(script_dir, job_name)
    raw_data_dir = job_dir / "raw_data"
    plots_opt_dir = job_dir / "plots_optimization"
    plots_obs_dir = job_dir / "plots_observables"
    ckpt_dir = job_dir / "checkpoints"
    for folder in (job_dir, raw_data_dir, plots_opt_dir, plots_obs_dir, ckpt_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if is_new_job:
        shutil.copy2(__file__, job_dir / "run_script.py")

    log_file = open(job_dir / "run.log", "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            # Written with Codex 03-02-26.
            self._streams = streams

        def write(self, data):
            # Written with Codex 03-02-26.
            for stream in self._streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            # Written with Codex 03-02-26.
            for stream in self._streams:
                stream.flush()

        def fileno(self):
            # Written with Codex 03-02-26.
            return self._streams[0].fileno()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"\n{'=' * 60}")
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Job dir:     {job_dir}")
    print(f"{'=' * 60}\n")

    history_path = raw_data_dir / "optimization_history.npz"
    checkpoint_path = job_dir / "vstate_variables.mpack"
    resume_checkpoint_path, checkpoint_resume_step = _select_resume_checkpoint(
        checkpoint_path,
        ckpt_dir,
    )

    graph, hi, ham = build_haldane_hamiltonian(
        Lx=lx,
        Ly=ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
        n_fermions=n_fermions,
    )

    assert n_fermions <= graph.n_nodes
    assert int(graph.n_nodes) == 2 * n_sc_cells, (
        f"Unexpected site count: n_sites={graph.n_nodes}, expected={2 * n_sc_cells}"
    )
    print(f"n_sites={graph.n_nodes}, n_fermions={hi.n_fermions}")
    print(f"supercell_matrix={supercell_matrix.tolist()}, n_sc_cells={n_sc_cells}")
    print(f"Lx={lx}, Ly={ly}, V1={V1}, max_conn_size={ham.max_conn_size}")

    positions, sc_t1, sc_t2, lattice = _build_geometry(supercell_matrix)
    sc_basis = np.stack([sc_t1, sc_t2], axis=0)

    e_ni = _noninteracting_energy(
        supercell_matrix,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        n_fermions=n_fermions,
    )
    print(f"Non-interacting reference energy (V1=0): {e_ni:.10f}")

    print("\nComputing Bloch-ED reference energies ...")
    ed_energies = _compute_haldane_ed_reference_energies(
        supercell_matrix=supercell_matrix,
        n_fermions=n_fermions,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
    )
    print()

    pair_classes_hash, pair_distances_hash, pair_classes_np, pair_distances_np = _make_pair_data(
        positions,
        sc_t1,
        sc_t2,
    )
    n_pair_classes = len(pair_distances_hash)
    print(f"Number of distinct pair-distance classes: {n_pair_classes}")
    print(f"displacement_only_attention={displacement_only_attention}")

    slater_initial_m_orbitals = None
    if slater_init_mode == "noninteracting":
        slater_initial_m_orbitals = noninteracting_slater_orbitals_haldane(
            graph=graph,
            Lx=lx,
            Ly=ly,
            n_fermions=n_fermions,
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
        )
        print("Initializing Slater determinant from non-interacting orbitals.")
    elif slater_init_mode != "random":
        raise ValueError(f"Unknown SLATER_INIT_MODE={slater_init_mode!r}")

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

    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        n_chains=n_chains,
        sweep_size=sweep_size,
    )

    if sample_type == "MC":
        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size,
        )
        ham_sr = ham
    elif sample_type == "FullSum":
        ham_sr = ham.to_fermionoperator2nd()
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        raise ValueError(f"Unknown sample_type={sample_type!r}")

    print(f"total # of wavefunction parameters: {nk.jax.tree_size(vstate.parameters)}")

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
            vstate.variables,
            resume_checkpoint_path.read_bytes(),
        )
        print(f"Loaded checkpoint: {resume_checkpoint_path}")
    elif start_step > 0:
        raise FileNotFoundError(
            "Expected checkpoint for resumption, but neither "
            f"{checkpoint_path} nor any step_*.mpack file was found."
        )

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        mode="complex",
    )

    base_callback = make_optimization_callback(job_dir)
    checkpoint_every = max(1, n_iter // 10)
    checkpoint_steps = set(range(checkpoint_every - 1, n_iter, checkpoint_every))

    def _callback(step: int, log_data: dict, driver_obj) -> bool:
        # Written with Codex 03-02-26.
        base_callback(step, log_data, driver_obj)
        if step in checkpoint_steps:
            abs_step = start_step + step + 1
            ckpt = ckpt_dir / f"step_{abs_step}.mpack"
            ckpt.write_bytes(flax.serialization.to_bytes(vstate.variables))
            print(f"[Checkpoint] Saved at absolute step {abs_step}: {ckpt}", flush=True)
        return True

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=_callback)

    checkpoint_path.write_bytes(flax.serialization.to_bytes(vstate.variables))
    param_leaves, param_treedef = jax.tree_util.tree_flatten(vstate.parameters)
    np.savez(job_dir / "vstate_parameters_leaves.npz", *[np.asarray(x) for x in param_leaves])
    (job_dir / "vstate_parameters_treedef.txt").write_text(repr(param_treedef) + "\n")

    end_step = start_step + n_iter
    log.serialize(job_dir / f"runtime_log_step{start_step}_to_step{end_step}")
    log.serialize(job_dir / "runtime_log")

    energy_hist = log.data.get("Energy", None)
    if energy_hist is None:
        iters_new = np.array([], dtype=np.int64)
        mean_new = np.array([], dtype=np.complex128)
        sigma_new = np.array([], dtype=np.float64)
        var_new = np.array([], dtype=np.float64)
        tau_new = np.array([], dtype=np.float64)
        rhat_new = np.array([], dtype=np.float64)
    else:
        iters_new = np.asarray(energy_hist.iters, dtype=np.int64) + start_step
        mean_new = np.asarray(energy_hist["Mean"], dtype=np.complex128)
        sigma_new = np.asarray(energy_hist["Sigma"], dtype=np.float64)
        var_new = np.asarray(energy_hist["Variance"], dtype=np.float64)
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
        un_iters_new = np.asarray(un_hist.iters, dtype=np.int64) + start_step
        un_values_new = np.asarray(un_hist, dtype=np.float64)
    else:
        un_iters_new = np.array([], dtype=np.int64)
        un_values_new = np.array([], dtype=np.float64)

    hist = {
        "iters": _concat(history_prev["iters"], iters_new),
        "energy_mean": _concat(history_prev["energy_mean"], mean_new),
        "energy_sigma": _concat(history_prev["energy_sigma"], sigma_new),
        "energy_variance": _concat(history_prev["energy_variance"], var_new),
        "energy_tau": _concat(history_prev["energy_tau"], tau_new),
        "energy_rhat": _concat(history_prev["energy_rhat"], rhat_new),
        "update_norm_iters": _concat(history_prev["update_norm_iters"], un_iters_new),
        "update_norm_values": _concat(history_prev["update_norm_values"], un_values_new),
    }
    hist["energy_std_local"] = np.sqrt(np.maximum(np.real(hist["energy_variance"]), 0.0))
    np.savez(history_path, **hist)

    iters = hist["iters"]
    energy_mean = hist["energy_mean"]
    energy_sigma = hist["energy_sigma"]
    energy_std_local = hist["energy_std_local"]
    energy_tau = hist["energy_tau"]
    energy_rhat = hist["energy_rhat"]
    un_iters = hist["update_norm_iters"]
    un_values = hist["update_norm_values"]
    total_steps = int(iters[-1]) + 1 if iters.size > 0 else start_step

    def _energy_band(ax, x, mean, sigma, **kwargs):
        # Written with Codex 03-02-26.
        ax.plot(x, np.real(mean), **kwargs)
        ax.fill_between(
            x,
            np.real(mean) - sigma,
            np.real(mean) + sigma,
            color=kwargs.get("color", "tab:blue"),
            alpha=0.2,
            linewidth=0,
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    _energy_band(ax, iters, energy_mean, energy_sigma, lw=1.8, color="tab:blue", label="VMC_SR")
    if V1 == 0.0:
        ax.axhline(e_ni, color="tab:purple", ls="--", lw=1.5, label=f"NI ref ({e_ni:.6f})")
    _add_ed_lines(ax, ed_energies)
    ax.set(
        title="Haldane model: ViT x Slater optimisation",
        xlabel="Optimisation step",
        ylabel="Energy",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    _save_fig(fig, plots_opt_dir / "energy_vs_step.png")

    i0 = max(int(math.floor(0.2 * iters.size)), 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    if iters.size > 0:
        _energy_band(
            ax,
            iters[i0:],
            energy_mean[i0:],
            energy_sigma[i0:],
            lw=1.8,
            color="tab:blue",
            label="VMC_SR",
        )
        if V1 == 0.0:
            ax.axhline(e_ni, color="tab:purple", ls="--", lw=1.5, label=f"NI ref ({e_ni:.6f})")
        _add_ed_lines(ax, ed_energies)
        ax.legend()
    ax.set(xlabel="Optimisation step", ylabel="Energy")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "energy_vs_step_final80pct.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.log10(np.maximum(energy_std_local, 1e-30)), color="tab:purple", lw=1.5)
    ax.set(
        title="Local energy standard deviation",
        xlabel="Optimisation step",
        ylabel="log10 Std(E_loc)",
    )
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "local_energy_std_log10.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if un_values.size > 0:
        ax.plot(un_iters, un_values, color="tab:red", lw=1.5)
        ax.set_yscale("log")
    ax.set(
        title="Update norm",
        xlabel="Optimisation step",
        ylabel=r"||$\Delta\theta$||$_2$",
    )
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "update_norm.png")

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

    shutil.copy2(job_dir / "run_script.py", plots_opt_dir / "run_script_used.py")

    if isinstance(vstate, nk.vqs.FullSumState):
        obs_n_samples_eff = min(obs_n_samples, 100_000)
        obs_n_discard_eff = obs_n_discard_per_chain
        obs_sampler = nk.sampler.MetropolisFermionHop(
            hi,
            graph=graph,
            n_chains=min(n_chains, 256),
            sweep_size=sweep_size,
        )
        obs_vstate = nk.vqs.MCState(
            obs_sampler,
            model,
            n_samples=obs_n_samples_eff,
            n_discard_per_chain=obs_n_discard_eff,
        )
        obs_vstate.variables = vstate.variables
        print(
            f"\nSampling observables (FullSumState->MCState) at step {total_steps}: "
            f"n_samples={obs_n_samples_eff}, n_discard={obs_n_discard_eff}"
        )
        samples_obs = np.asarray(obs_vstate.sample()).reshape(-1, hi.size).astype(np.float64)
    else:
        obs_n_samples_eff = obs_n_samples
        obs_n_discard_eff = obs_n_discard_per_chain
        print(
            f"\nSampling observables at step {total_steps}: "
            f"n_samples={obs_n_samples_eff}, n_discard={obs_n_discard_eff}"
        )
        samples_obs = (
            np.asarray(
                vstate.sample(
                    n_samples=obs_n_samples_eff,
                    n_discard_per_chain=obs_n_discard_eff,
                )
            )
            .reshape(-1, hi.size)
            .astype(np.float64)
        )

    pc = pair_correlation_cartesian(
        samples=samples_obs,
        positions=positions,
        basis_vectors=sc_basis,
        pbc=np.array([True, True]),
        Lx=1,
        Ly=1,
        n_fermions=n_fermions,
    )
    charge_density = pc["charge_density"]
    corr_matrix = pc["corr_matrix"]
    r_values_plot = pc["r_values_plot"]
    g_r_plot = pc["g_r_plot"]

    rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(
        positions=positions,
        basis_coords=np.asarray(lattice.site_to_cell),
        translations=pc["translations"],
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
    )

    q_list_raw, q_list = _supercell_q_vectors(lattice)
    s_q = _structure_factor(samples_obs, positions, q_list, n_fermions)
    q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot = radial_average_structure_factor(
        q_list,
        s_q,
        set_q0_to_zero=True,
    )

    positions_folded = _fold_positions(positions, sc_t1, sc_t2)

    step_suffix = f"_step{total_steps}"
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
        obs_n_discard_per_chain=np.array([obs_n_discard_eff], dtype=np.int64),
        optimization_step=np.array([total_steps], dtype=np.int64),
        sc_basis=sc_basis,
        sc_t1=sc_t1,
        sc_t2=sc_t2,
        e_ni=np.array([e_ni]),
        pair_classes=pair_classes_np,
        pair_distances=pair_distances_np,
    )

    rhombus = np.array([np.zeros(2), sc_t1, sc_t1 + sc_t2, sc_t2, np.zeros(2)])
    rhombus_centered = np.array([
        -0.5 * sc_t1 - 0.5 * sc_t2,
        0.5 * sc_t1 - 0.5 * sc_t2,
        0.5 * sc_t1 + 0.5 * sc_t2,
        -0.5 * sc_t1 + 0.5 * sc_t2,
        -0.5 * sc_t1 - 0.5 * sc_t2,
    ])

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sc_plot = ax.scatter(
        positions_folded[:, 0],
        positions_folded[:, 1],
        c=charge_density,
        s=140,
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.max(charge_density)),
    )
    ax.plot(rhombus[:, 0], rhombus[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title=r"Charge density $\langle n_i \rangle$", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(sc_plot, ax=ax, label=r"$\langle n_i \rangle$")
    _save_fig(fig, plots_obs_dir / f"charge_density{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax.set(
        title="Pair correlation (minimum-image distances)",
        xlabel="Distance r (minimum image)",
        ylabel="g(r)",
    )
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"pair_correlation_radial{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc2 = ax.scatter(rel_min[:, 0], rel_min[:, 1], c=g_site, s=180, cmap="viridis")
    ax.scatter([0.0], [0.0], s=220, facecolors="none", edgecolors="black", linewidths=1.1)
    ax.plot(rhombus_centered[:, 0], rhombus_centered[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(
        title="Pair correlation mapped to 2D (minimum image)",
        xlabel="x (minimum image)",
        ylabel="y (minimum image)",
        aspect="equal",
    )
    fig.colorbar(sc2, ax=ax, label="g(r)")
    _save_fig(fig, plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(
        q_shell_unique,
        s_q_abs,
        yerr=s_q_abs_err,
        fmt="o-",
        ms=4,
        lw=1.2,
        capsize=2,
        color="tab:blue",
    )
    ax.set(
        title="Static structure factor (radial, WS-BZ folded)",
        xlabel=r"$|\mathbf{q}|$ (WS-BZ)",
        ylabel="S(q)",
    )
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"static_structure_factor{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc3 = ax.scatter(
        q_list[:, 0],
        q_list[:, 1],
        c=s_q_plot.real,
        s=120,
        cmap="magma",
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set(
        title="Static structure factor in q-space (WS-BZ folded)",
        xlabel=r"$q_x$",
        ylabel=r"$q_y$",
        aspect="equal",
    )
    ax.grid(alpha=0.2)
    fig.colorbar(sc3, ax=ax, label=r"$S(\mathbf{q})$")
    _save_fig(fig, plots_obs_dir / f"static_structure_factor_qspace{step_suffix}.png")

    shutil.copy2(job_dir / "run_script.py", plots_obs_dir / f"run_script_used{step_suffix}.py")

    summary = {
        "job_dir": str(job_dir),
        "backend": backend,
        "model_type": "vit_slater",
        "sample_type": sample_type,
        "system": {
            "supercell_matrix": supercell_matrix.tolist(),
            "n_supercell_cells": n_sc_cells,
            "Lx": lx,
            "Ly": ly,
            "n_sites": int(graph.n_nodes),
            "n_fermions": n_fermions,
            "V1": V1,
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "m": m,
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
            "checkpoint_every": checkpoint_every,
            "chunk_size": chunk_size,
        },
        "observables": {
            "obs_n_samples": int(samples_obs.shape[0]),
            "obs_n_discard_per_chain": int(obs_n_discard_eff),
            "optimization_step": total_steps,
        },
        "n_wavefunction_params": int(nk.jax.tree_size(vstate.parameters)),
        "final_energy_real": float(np.real(energy_mean[-1])) if energy_mean.size > 0 else None,
        "final_energy_sigma": float(energy_sigma[-1]) if energy_sigma.size > 0 else None,
        "final_update_norm": float(un_values[-1]) if un_values.size > 0 else None,
        "e_ni": e_ni,
        "error_vs_ni": (
            float(np.real(energy_mean[-1])) - e_ni if energy_mean.size > 0 else None
        ),
        "ed_reference_energies": {
            str(n_bands): (None if e_val is None else float(e_val))
            for n_bands, e_val in ed_energies.items()
        },
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nNon-interacting ref energy = {e_ni:.10f}")
    if energy_mean.size > 0:
        print(f"Final energy (Re)          = {np.real(energy_mean[-1]):.10f}")
        print(f"Error vs NI ref            = {float(np.real(energy_mean[-1])) - e_ni:.6e}")
    if un_values.size > 0:
        print(f"Final update norm          = {float(un_values[-1]):.6e}")
    if energy_std_local.size > 0:
        print(f"Final std(E_loc)           = {float(energy_std_local[-1]):.10f}")
    for n_bands, e_val in sorted(ed_energies.items()):
        tag = f"{e_val:.10f}" if e_val is not None else "skipped"
        print(f"ED ref ({n_bands} Band)            = {tag}")
    print(f"Observables raw:    {obs_raw_path}")
    print(f"Optimisation plots: {plots_opt_dir}")
    print(f"Observable plots:   {plots_obs_dir}")


if __name__ == "__main__":
    main()
