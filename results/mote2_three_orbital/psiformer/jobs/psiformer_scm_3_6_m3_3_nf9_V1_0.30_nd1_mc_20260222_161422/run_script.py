"""PsiFormer parent run script — MoTe2 three-orbital model.

Running this script creates a timestamped job folder under `jobs/`, copies
itself there as `run_script.py`, and executes VMC optimisation followed by
observable collection.  Re-running `run_script.py` from inside its own job
folder resumes an interrupted run (checkpoint loading is automatic).

Directory layout after one run
-------------------------------
results/mote2_three_orbital/psiformer/
├── run.py                          ← this file (never modified by a run)
└── jobs/
    └── psiformer_scm_3_6_m3_3_nf9_V1_0.30_nd1_mc_YYYYMMDD_HHMMSS/
        ├── run_script.py           ← exact copy of run.py at launch time
        ├── run.log                 ← stdout + stderr
        ├── summary.json
        ├── vstate_variables.mpack
        ├── vstate_parameters_leaves.npz
        ├── vstate_parameters_treedef.txt
        ├── runtime_log.json
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
  N_DISCARD_PER_CHAIN         Discard per chain per step (default: 8)
  SWEEP_SIZE                  MCMC sweep size (default: 2*n_fermions)
  N_CHAINS                    Number of MCMC chains (default: 512)
  LEARNING_RATE               SGD learning rate (default: 0.05)
  DIAG_SHIFT                  SR diagonal shift (default: 0.01)
  OBS_N_SAMPLES               Samples for post-training observables (default: 10×N_SAMPLES)
  OBS_N_DISCARD_PER_CHAIN     Discard for observable sampling (default: max(64, N_DISCARD))
  NUM_LAYERS                  Encoder depth (default: 2)
  D_MODEL                     Token embedding dimension (default: 32)
  N_HEADS                     Attention heads (default: 2)
  MLP_HIDDEN_FACTOR           MLP hidden-layer factor (default: 2)
  N_DETERMINANTS              Number of Slater determinants (default: 1)
  USE_JASTROW                 0/1 — add two-body Jastrow factor (default: 0)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import netket as nk
import numpy as np

from aidan_custom.bloch_ed import build_mote2_three_orbital_lattice_embedding
from aidan_custom.geometry import fold_to_shortest_k
from aidan_custom.models import LogPsiFormer
from aidan_custom.mote2_three_orbital import (
    MOTE2_A1,
    MOTE2_A2,
    mote2_three_orbital_reciprocal_vectors,
)
from aidan_custom.mote2_three_orbital_model import build_mote2_three_orbital_hamiltonian
from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
)
from aidan_custom.optimization import make_optimization_callback


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

def _build_geometry(supercell_matrix: np.ndarray, a_m: float):
    """Return site positions and supercell Bravais vectors."""
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
        positions[site] = float(bravais[0]) * a1 + float(bravais[1]) * a2 + offsets[int(orb)]

    sc_t1 = (
        float(lattice.supercell_matrix[0, 0]) * a1
        + float(lattice.supercell_matrix[1, 0]) * a2
    )
    sc_t2 = (
        float(lattice.supercell_matrix[0, 1]) * a1
        + float(lattice.supercell_matrix[1, 1]) * a2
    )
    return positions, sc_t1, sc_t2, lattice


def _supercell_reciprocal_vectors(sc_t1: np.ndarray, sc_t2: np.ndarray) -> np.ndarray:
    sc_mat = np.stack([sc_t1, sc_t2], axis=0)
    return 2.0 * np.pi * np.linalg.inv(sc_mat.T)


def _supercell_q_vectors(lattice, a_m: float):
    """BZ q-points of the supercell, folded to the MoTe2 primitive BZ."""
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    b1 = np.asarray(b1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    q_raw = np.array([
        lattice.kpoint_coefficients[kx, ky][0] * b1
        + lattice.kpoint_coefficients[kx, ky][1] * b2
        for kx in range(lattice.Lx)
        for ky in range(lattice.Ly)
    ])
    q_folded = np.array([
        fold_to_shortest_k(q, b1, b2, search_radius=8) for q in q_raw
    ])
    return q_raw, q_folded


def _hashable_matrix(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def _fold_positions(positions: np.ndarray, sc_t1: np.ndarray, sc_t2: np.ndarray):
    """Fold positions into the supercell rhombus (fractional coords in [0, 1))."""
    sc_cols = np.column_stack([sc_t1, sc_t2])
    frac = np.linalg.solve(sc_cols, positions.T).T
    frac_folded = frac - np.floor(frac)
    return (sc_cols @ frac_folded.T).T


# ---------------------------------------------------------------------------
# Observables
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

_HIST_DTYPES = {
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
    """Return (job_dir, is_new_job).

    If `raw_data/` already exists alongside the script (i.e. the script is
    running from inside a job folder), treat the script's directory as the
    job folder and resume.  Otherwise create a fresh job folder.
    """
    if (script_dir / "raw_data").exists():
        return script_dir, False
    job_dir = script_dir / "jobs" / job_name
    return job_dir, True


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

    # Physical system
    supercell_matrix = np.array([[3, 6], [-3, 3]], dtype=np.int64)
    n_fermions = 9
    V1 = _env_float("V1", 0.3)

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
    sample_type         = _env_str("SAMPLE_TYPE", "MC")
    n_iter              = _env_int("N_ITER", 5000)
    n_samples           = _env_int("N_SAMPLES", 1024 * 4)
    n_discard_per_chain = _env_int("N_DISCARD_PER_CHAIN", 8)
    sweep_size          = _env_int("SWEEP_SIZE", 2 * n_fermions)
    n_chains            = _env_int("N_CHAINS", 512)
    learning_rate       = _env_float("LEARNING_RATE", 0.05)
    diag_shift          = _env_float("DIAG_SHIFT", 0.01)

    # Post-training observables
    obs_n_samples           = _env_int("OBS_N_SAMPLES", n_samples * 10)
    obs_n_discard_per_chain = _env_int("OBS_N_DISCARD_PER_CHAIN", max(64, n_discard_per_chain))

    # PsiFormer architecture
    num_layers        = _env_int("NUM_LAYERS", 2)
    d_model           = _env_int("D_MODEL", 32)
    n_heads           = _env_int("N_HEADS", 2)
    mlp_hidden_factor = _env_int("MLP_HIDDEN_FACTOR", 2)
    n_determinants    = _env_int("N_DETERMINANTS", 1)
    use_jastrow       = _env_bool("USE_JASTROW", False)

    # -----------------------------------------------------------------------
    # Resolve job folder
    # -----------------------------------------------------------------------

    script_dir = Path(__file__).resolve().parent
    n_sc_cells = abs(int(np.linalg.det(supercell_matrix).round()))
    a, b = int(supercell_matrix[0, 0]), int(supercell_matrix[0, 1])
    c, d = int(supercell_matrix[1, 0]), int(supercell_matrix[1, 1])
    c_str = f"m{abs(c)}" if c < 0 else str(c)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = (
        f"psiformer_scm_{a}_{b}_{c_str}_{d}_nf{n_fermions}"
        f"_V1_{V1:.2f}_nd{n_determinants}_{sample_type.lower()}_{timestamp}"
    )

    job_dir, is_new_job = _resolve_job_dir(script_dir, job_name)
    raw_data_dir  = job_dir / "raw_data"
    plots_opt_dir = job_dir / "plots_optimization"
    plots_obs_dir = job_dir / "plots_observables"
    for folder in (job_dir, raw_data_dir, plots_opt_dir, plots_obs_dir):
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
    g_vectors = _supercell_reciprocal_vectors(sc_t1, sc_t2)
    sc_basis  = np.stack([sc_t1, sc_t2], axis=0)  # (2, 2) supercell basis

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = LogPsiFormer(
        hilbert=hi,
        positions=_hashable_matrix(positions),
        g_vectors=_hashable_matrix(g_vectors),
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        mlp_hidden_factor=mlp_hidden_factor,
        n_determinants=n_determinants,
        use_jastrow=use_jastrow,
        param_dtype=jnp.float64,
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
    if history_prev["iters"].size > 0:
        start_step = int(history_prev["iters"][-1]) + 1
        print(f"Resuming from step {start_step} ({history_prev['iters'].size} steps in history).")
    else:
        start_step = 0
        print("Starting fresh from step 0.")

    if checkpoint_path.exists():
        vstate.variables = flax.serialization.from_bytes(
            vstate.variables, checkpoint_path.read_bytes()
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
    elif start_step > 0:
        raise FileNotFoundError(
            f"Expected checkpoint for resumption at {checkpoint_path}, but none was found."
        )

    # -----------------------------------------------------------------------
    # Optimise
    # -----------------------------------------------------------------------

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham_sr, optimizer, variational_state=vstate, diag_shift=diag_shift, mode="complex",
    )
    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=make_optimization_callback(job_dir))

    # -----------------------------------------------------------------------
    # Save checkpoint and parameters
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
        iters_new        = np.array([], dtype=np.int64)
        mean_new         = np.array([], dtype=np.complex128)
        sigma_new        = np.array([], dtype=np.float64)
        var_new          = np.array([], dtype=np.float64)
        tau_new          = np.array([], dtype=np.float64)
        rhat_new         = np.array([], dtype=np.float64)
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

    iters             = hist["iters"]
    energy_mean       = hist["energy_mean"]
    energy_sigma      = hist["energy_sigma"]
    energy_std_local  = hist["energy_std_local"]
    energy_tau        = hist["energy_tau"]
    energy_rhat       = hist["energy_rhat"]
    un_iters          = hist["update_norm_iters"]
    un_values         = hist["update_norm_values"]
    total_steps       = int(iters[-1]) + 1 if iters.size > 0 else start_step

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
    ax.set(title="MoTe2 three-orbital: PsiFormer optimisation",
           xlabel="Optimisation step", ylabel="Energy")
    ax.grid(alpha=0.25); ax.legend()
    _save_fig(fig, plots_opt_dir / "energy_vs_step.png")

    # Final 80%
    i0 = max(int(math.floor(0.2 * iters.size)), 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    if iters.size > 0:
        _energy_band(ax, iters[i0:], energy_mean[i0:], energy_sigma[i0:],
                     lw=1.8, color="tab:blue", label="VMC_SR")
        ax.legend()
    ax.set(xlabel="Optimisation step", ylabel="Energy"); ax.grid(alpha=0.25)
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
    ax.set(title="Sampling diagnostics", xlabel="Optimisation step"); ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "sampling_diagnostics.png")

    # -----------------------------------------------------------------------
    # Post-training observables
    # -----------------------------------------------------------------------

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

    # Pair correlation (Lx=Ly=1 with supercell as the simulation cell)
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
    corr_matrix    = pc["corr_matrix"]
    r_values_plot  = pc["r_values_plot"]
    g_r_plot       = pc["g_r_plot"]

    rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(
        positions=positions,
        basis_coords=np.asarray(lattice.site_to_cell),
        translations=pc["translations"],
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
    )

    # Structure factor
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
    sc = ax.scatter(positions_folded[:, 0], positions_folded[:, 1],
                    c=charge_density, s=140, cmap="viridis",
                    vmin=0.0, vmax=float(np.max(charge_density)))
    ax.plot(rhombus[:, 0], rhombus[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title=r"Charge density $\langle n_i \rangle$",
           xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(sc, ax=ax, label=r"$\langle n_i \rangle$")
    _save_fig(fig, plots_obs_dir / f"charge_density{step_suffix}.png")

    # Pair correlation — radial
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax.set(title="Pair correlation", xlabel="Distance r", ylabel="g(r)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"pair_correlation_radial{step_suffix}.png")

    # Pair correlation — 2D minimum image
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc2 = ax.scatter(rel_min[:, 0], rel_min[:, 1], c=g_site, s=180, cmap="viridis")
    ax.scatter([0.0], [0.0], s=220, facecolors="none", edgecolors="black", linewidths=1.1)
    ax.plot(rhombus_centered[:, 0], rhombus_centered[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title="Pair correlation mapped to 2D",
           xlabel="x (minimum image)", ylabel="y (minimum image)", aspect="equal")
    fig.colorbar(sc2, ax=ax, label="g(r)")
    _save_fig(fig, plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png")

    # Structure factor — radial
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(q_shell_unique, s_q_abs, yerr=s_q_abs_err,
                fmt="o-", ms=4, lw=1.2, capsize=2, color="tab:blue")
    ax.set(title="Static structure factor (radial)",
           xlabel=r"$|\mathbf{q}|$", ylabel="S(q)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"static_structure_factor{step_suffix}.png")

    # Structure factor — q-space map
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc3 = ax.scatter(q_list[:, 0], q_list[:, 1], c=s_q_plot.real,
                     s=120, cmap="magma", edgecolors="black", linewidths=0.25)
    ax.set(title="Static structure factor in q-space",
           xlabel=r"$q_x$", ylabel=r"$q_y$", aspect="equal")
    ax.grid(alpha=0.2)
    fig.colorbar(sc3, ax=ax, label=r"$S(\mathbf{q})$")
    _save_fig(fig, plots_obs_dir / f"static_structure_factor_qspace{step_suffix}.png")

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------

    summary = {
        "job_dir":    str(job_dir),
        "backend":    backend,
        "model_type": "psiformer",
        "sample_type": sample_type,
        "system": {
            "supercell_matrix": supercell_matrix.tolist(),
            "n_supercell_cells": n_sc_cells,
            "n_sites": int(graph.n_nodes),
            "n_fermions": n_fermions,
            "V1": V1, "delta": delta, "ez": ez,
            "t_th1": t_th1, "t_hh1": t_hh1, "t_th2": t_th2,
            "t_hh3": t_hh3, "t_tt1": t_tt1, "a_m": a_m, "ph_conj": ph_conj,
        },
        "network": {
            "num_layers": num_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "mlp_hidden_factor": mlp_hidden_factor,
            "n_determinants": n_determinants,
            "use_jastrow": use_jastrow,
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
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nFinal energy (Re)   = {np.real(energy_mean[-1]):.10f}")
    if un_values.size > 0:
        print(f"Final update norm   = {float(un_values[-1]):.6e}")
    print(f"Final std(E_loc)    = {float(energy_std_local[-1]):.10f}")
    print(f"Observables raw:    {obs_raw_path}")
    print(f"Optimisation plots: {plots_opt_dir}")
    print(f"Observable plots:   {plots_obs_dir}")


if __name__ == "__main__":
    main()
