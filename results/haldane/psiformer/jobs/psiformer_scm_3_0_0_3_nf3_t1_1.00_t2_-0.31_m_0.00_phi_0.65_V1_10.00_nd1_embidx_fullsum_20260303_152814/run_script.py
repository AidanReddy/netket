"""PsiFormer parent run script for the Haldane model."""

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
from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import LogPsiFormer
from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
)
from aidan_custom.optimization import make_optimization_callback


_ED_COLORS = {1: "tab:orange", 2: "tab:green"}
_ED_LS = {1: "--", 2: "-."}
_HIST_DTYPES = {
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


def _env_bool(key: str, default: bool) -> bool:
    # Written with Codex 03-02-26.
    return bool(int(os.environ.get(key, str(int(default)))))


def _parse_supercell_matrix(raw: str) -> np.ndarray:
    # Written with Codex 03-02-26.
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError(f"SUPERCELL_MATRIX must have 4 ints, got {raw!r}.")
    scm = np.asarray(vals, dtype=np.int64).reshape(2, 2)
    if int(round(np.linalg.det(scm))) == 0:
        raise ValueError(f"SUPERCELL_MATRIX must be invertible, got {scm.tolist()}.")
    return scm


def _diagonal_shape_from_supercell(supercell_matrix: np.ndarray) -> tuple[int, int]:
    # Written with Codex 03-02-26.
    scm = np.asarray(supercell_matrix, dtype=np.int64)
    if scm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 supercell matrix, got {scm.shape}.")
    if int(scm[0, 1]) != 0 or int(scm[1, 0]) != 0:
        raise NotImplementedError(f"Only diagonal rectangular Haldane supercells are supported, got {scm.tolist()}.")
    lx = int(scm[0, 0])
    ly = int(scm[1, 1])
    if lx <= 0 or ly <= 0:
        raise ValueError(f"Diagonal entries must be positive, got {scm.tolist()}.")
    return lx, ly


def _build_geometry(supercell_matrix: np.ndarray):
    # Written with Codex 03-02-26.
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    lattice = build_haldane_lattice_embedding(Lx=lx, Ly=ly)
    a1 = np.asarray(PRIMITIVE_A1, dtype=np.float64)
    a2 = np.asarray(PRIMITIVE_A2, dtype=np.float64)
    offsets = np.array([np.asarray(SUBLATTICE_A_OFFSET), np.asarray(SUBLATTICE_B_OFFSET)], dtype=np.float64)
    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell
    positions = np.empty((n_sites, 2), dtype=np.float64)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        bravais = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = float(bravais[0]) * a1 + float(bravais[1]) * a2 + offsets[int(orb)]
    sc_t1 = float(lattice.supercell_matrix[0, 0]) * a1 + float(lattice.supercell_matrix[1, 0]) * a2
    sc_t2 = float(lattice.supercell_matrix[0, 1]) * a1 + float(lattice.supercell_matrix[1, 1]) * a2
    return positions, sc_t1, sc_t2, lattice


def _supercell_reciprocal_vectors(sc_t1: np.ndarray, sc_t2: np.ndarray) -> np.ndarray:
    # Written with Codex 03-02-26.
    sc_mat = np.stack([sc_t1, sc_t2], axis=0)
    return 2.0 * np.pi * np.linalg.inv(sc_mat.T)


def _supercell_q_vectors(lattice) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 03-02-26.
    b1, b2 = reciprocal_vectors(np.asarray(PRIMITIVE_A1), np.asarray(PRIMITIVE_A2))
    q_raw = np.array(
        [
            lattice.kpoint_coefficients[kx, ky][0] * b1
            + lattice.kpoint_coefficients[kx, ky][1] * b2
            for kx in range(lattice.Lx)
            for ky in range(lattice.Ly)
        ],
        dtype=np.float64,
    )
    q_folded = np.array([fold_to_shortest_k(q, b1, b2, search_radius=8) for q in q_raw], dtype=np.float64)
    return q_raw, q_folded


def _hashable_matrix(arr: np.ndarray):
    # Written with Codex 03-02-26.
    arr = np.asarray(arr, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def _fold_positions(positions: np.ndarray, sc_t1: np.ndarray, sc_t2: np.ndarray):
    # Written with Codex 03-02-26.
    sc_cols = np.column_stack([sc_t1, sc_t2])
    frac = np.linalg.solve(sc_cols, positions.T).T
    frac_folded = frac - np.floor(frac)
    return (sc_cols @ frac_folded.T).T


def _compute_haldane_ed_reference_energies(supercell_matrix: np.ndarray, n_fermions: int, *, t1: float, t2: float, phi: float, m: float, V1: float, dim_cutoff: int = 100_000) -> dict[int, float | None]:
    # Written with Codex 03-02-26.
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    out: dict[int, float | None] = {}
    for n_bands in (1, 2):
        proj_ham = build_haldane_projected_hamiltonian(Lx=lx, Ly=ly, t1=t1, t2=t2, phi=phi, m=m, selected_bands=tuple(range(n_bands)), V1=V1)
        n_orbitals = int(proj_ham.one_body.shape[0])
        lattice = proj_ham.lattice
        total_dim = math.comb(n_orbitals, n_fermions)
        if total_dim > dim_cutoff * lattice.Lx * lattice.Ly:
            out[n_bands] = None
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
            out[n_bands] = None
            continue
        sector_results = solve_projected_all_momentum_sectors(projected_hamiltonian=proj_ham, n_particles=n_fermions)
        all_eigs = [res["eigenvalues"] for res in sector_results.values() if len(res["eigenvalues"]) > 0]
        out[n_bands] = None if not all_eigs else float(min(float(ev[0]) for ev in all_eigs))
    return out


def _add_ed_lines(ax, ed_energies: dict[int, float | None]) -> None:
    # Written with Codex 03-02-26.
    for n_bands in (1, 2):
        e_val = ed_energies.get(n_bands)
        if e_val is None:
            continue
        ax.axhline(e_val, color=_ED_COLORS[n_bands], ls=_ED_LS[n_bands], lw=1.5, label=f"{n_bands} Band", zorder=2)


def _structure_factor(samples: np.ndarray, positions: np.ndarray, q_vectors: np.ndarray, n_fermions: int) -> np.ndarray:
    # Written with Codex 03-02-26.
    phases = np.exp(-1j * (positions @ q_vectors.T))
    rho_q = samples @ phases
    return np.mean(np.abs(rho_q) ** 2, axis=0) / n_fermions


def _load_history(path: Path) -> dict:
    # Written with Codex 03-02-26.
    hist = {k: np.array([], dtype=v) for k, v in _HIST_DTYPES.items()}
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        for k, dtype in _HIST_DTYPES.items():
            if k in loaded:
                hist[k] = np.asarray(loaded[k], dtype=dtype)
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
        raise RuntimeError(f"Expected GPU backend, got {backend!r}.")

    scm_str = os.environ.get("SUPERCELL_MATRIX", "")
    supercell_matrix = _parse_supercell_matrix(scm_str) if scm_str else np.array([[3, 0], [0, 3]], dtype=np.int64)
    lx, ly = _diagonal_shape_from_supercell(supercell_matrix)
    n_fermions = _env_int("N_FERMIONS", 3)
    V1 = _env_float("V1", 10.0)
    t1 = _env_float("T1", 1.0)
    phi = _env_float("PHI", 0.65)
    t2 = _env_float("T2", -1.0 / (4.0 * math.cos(phi)))
    m = _env_float("M", 0.0)

    sample_type = _env_str("SAMPLE_TYPE", "FullSum")
    n_iter = _env_int("N_ITER", 2_000)
    n_samples = _env_int("N_SAMPLES", 1024 * 4)
    n_discard_per_chain = _env_int("N_DISCARD_PER_CHAIN", 8)
    sweep_size = _env_int("SWEEP_SIZE", 2 * n_fermions)
    n_chains = _env_int("N_CHAINS", 512)
    learning_rate = _env_float("LEARNING_RATE", 0.05)
    diag_shift = _env_float("DIAG_SHIFT", 0.01)
    obs_n_samples = _env_int("OBS_N_SAMPLES", 10 * n_samples)
    obs_n_discard_per_chain = _env_int("OBS_N_DISCARD_PER_CHAIN", max(64, n_discard_per_chain))

    num_layers = _env_int("NUM_LAYERS", 2)
    d_model = _env_int("D_MODEL", 32)
    n_heads = _env_int("N_HEADS", 2)
    mlp_hidden_factor = _env_int("MLP_HIDDEN_FACTOR", 2)
    n_determinants = _env_int("N_DETERMINANTS", 1)
    use_jastrow = _env_bool("USE_JASTROW", False)
    embedding_type = _env_str("EMBEDDING_TYPE", "site_index")
    if embedding_type not in ("periodic", "site_index"):
        raise ValueError(f"Invalid EMBEDDING_TYPE={embedding_type!r}")

    script_dir = Path(__file__).resolve().parent
    n_sc_cells = abs(int(np.linalg.det(supercell_matrix).round()))
    a, b = int(supercell_matrix[0, 0]), int(supercell_matrix[0, 1])
    c, d = int(supercell_matrix[1, 0]), int(supercell_matrix[1, 1])
    c_str = f"m{abs(c)}" if c < 0 else str(c)
    emb_tag = "per" if embedding_type == "periodic" else "idx"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = (
        f"psiformer_scm_{a}_{b}_{c_str}_{d}_nf{n_fermions}"
        f"_t1_{t1:.2f}_t2_{t2:.2f}_m_{m:.2f}_phi_{phi:.2f}"
        f"_V1_{V1:.2f}_nd{n_determinants}_emb{emb_tag}_{sample_type.lower()}_{timestamp}"
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
            for s in self._streams:
                s.write(data)
                s.flush()
        def flush(self):
            # Written with Codex 03-02-26.
            for s in self._streams:
                s.flush()
        def fileno(self):
            # Written with Codex 03-02-26.
            return self._streams[0].fileno()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    history_path = raw_data_dir / "optimization_history.npz"
    checkpoint_path = job_dir / "vstate_variables.mpack"
    resume_checkpoint_path, checkpoint_resume_step = _select_resume_checkpoint(
        checkpoint_path,
        ckpt_dir,
    )

    graph, hi, ham = build_haldane_hamiltonian(Lx=lx, Ly=ly, t1=t1, t2=t2, phi=phi, m=m, V1=V1, n_fermions=n_fermions)
    assert int(graph.n_nodes) == 2 * n_sc_cells
    positions, sc_t1, sc_t2, lattice = _build_geometry(supercell_matrix)
    g_vectors = _supercell_reciprocal_vectors(sc_t1, sc_t2)
    sc_basis = np.stack([sc_t1, sc_t2], axis=0)

    ed_energies = _compute_haldane_ed_reference_energies(supercell_matrix, n_fermions, t1=t1, t2=t2, phi=phi, m=m, V1=V1)

    model = LogPsiFormer(
        hilbert=hi,
        positions=_hashable_matrix(positions),
        g_vectors=_hashable_matrix(g_vectors),
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        mlp_hidden_factor=mlp_hidden_factor,
        n_determinants=n_determinants,
        embedding_type=embedding_type,
        use_jastrow=use_jastrow,
        param_dtype=jnp.float64,
    )

    sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=n_chains, sweep_size=sweep_size)
    if sample_type == "MC":
        vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
        ham_sr = ham
    elif sample_type == "FullSum":
        ham_sr = ham.to_fermionoperator2nd()
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        raise ValueError(f"Unknown sample_type={sample_type!r}")

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
    driver = nk.driver.VMC_SR(ham_sr, optimizer, variational_state=vstate, diag_shift=diag_shift, mode="complex")
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
        iters_new = np.array([], dtype=np.int64); mean_new = np.array([], dtype=np.complex128); sigma_new = np.array([], dtype=np.float64); var_new = np.array([], dtype=np.float64); tau_new = np.array([], dtype=np.float64); rhat_new = np.array([], dtype=np.float64)
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
        un_iters_new = np.array([], dtype=np.int64); un_values_new = np.array([], dtype=np.float64)
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
    iters = hist["iters"]; energy_mean = hist["energy_mean"]; energy_sigma = hist["energy_sigma"]; energy_std_local = hist["energy_std_local"]; energy_tau = hist["energy_tau"]; energy_rhat = hist["energy_rhat"]; un_iters = hist["update_norm_iters"]; un_values = hist["update_norm_values"]
    total_steps = int(iters[-1]) + 1 if iters.size > 0 else start_step

    def _energy_band(ax, x, mean, sigma, **kw):
        # Written with Codex 03-02-26.
        ax.plot(x, np.real(mean), **kw)
        ax.fill_between(x, np.real(mean) - sigma, np.real(mean) + sigma, color=kw.get("color", "tab:blue"), alpha=0.2, linewidth=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    _energy_band(ax, iters, energy_mean, energy_sigma, lw=1.8, color="tab:blue", label="VMC_SR")
    _add_ed_lines(ax, ed_energies)
    ax.set(title="Haldane model: PsiFormer optimisation", xlabel="Optimisation step", ylabel="Energy")
    ax.grid(alpha=0.25); ax.legend()
    _save_fig(fig, plots_opt_dir / "energy_vs_step.png")

    i0 = max(int(math.floor(0.2 * iters.size)), 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    if iters.size > 0:
        _energy_band(ax, iters[i0:], energy_mean[i0:], energy_sigma[i0:], lw=1.8, color="tab:blue", label="VMC_SR")
        _add_ed_lines(ax, ed_energies); ax.legend()
    ax.set(xlabel="Optimisation step", ylabel="Energy"); ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "energy_vs_step_final80pct.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.log10(np.maximum(energy_std_local, 1e-30)), color="tab:purple", lw=1.5)
    ax.set(title="Local energy standard deviation", xlabel="Optimisation step", ylabel="log10 Std(E_loc)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "local_energy_std_log10.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if un_values.size > 0:
        ax.plot(un_iters, un_values, color="tab:red", lw=1.5); ax.set_yscale("log")
    ax.set(title="Update norm", xlabel="Optimisation step", ylabel=r"||$\Delta\theta$||$_2$")
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
    ax.set(title="Sampling diagnostics", xlabel="Optimisation step"); ax.grid(alpha=0.25)
    _save_fig(fig, plots_opt_dir / "sampling_diagnostics.png")
    shutil.copy2(job_dir / "run_script.py", plots_opt_dir / "run_script_used.py")

    if isinstance(vstate, nk.vqs.FullSumState):
        obs_n_samples_eff = min(obs_n_samples, 100_000)
        obs_sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=min(n_chains, 256), sweep_size=sweep_size)
        obs_vstate = nk.vqs.MCState(obs_sampler, model, n_samples=obs_n_samples_eff, n_discard_per_chain=obs_n_discard_per_chain)
        obs_vstate.variables = vstate.variables
        samples_obs = np.asarray(obs_vstate.sample()).reshape(-1, hi.size).astype(np.float64)
        obs_discard_eff = obs_n_discard_per_chain
    else:
        obs_n_samples_eff = obs_n_samples
        samples_obs = np.asarray(vstate.sample(n_samples=obs_n_samples, n_discard_per_chain=obs_n_discard_per_chain)).reshape(-1, hi.size).astype(np.float64)
        obs_discard_eff = obs_n_discard_per_chain

    pc = pair_correlation_cartesian(samples=samples_obs, positions=positions, basis_vectors=sc_basis, pbc=np.array([True, True]), Lx=1, Ly=1, n_fermions=n_fermions)
    charge_density = pc["charge_density"]; corr_matrix = pc["corr_matrix"]; r_values_plot = pc["r_values_plot"]; g_r_plot = pc["g_r_plot"]
    rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(positions=positions, basis_coords=np.asarray(lattice.site_to_cell), translations=pc["translations"], r_values_plot=r_values_plot, g_r_plot=g_r_plot)
    q_list_raw, q_list = _supercell_q_vectors(lattice)
    s_q = _structure_factor(samples_obs, positions, q_list, n_fermions)
    q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot = radial_average_structure_factor(q_list, s_q, set_q0_to_zero=True)
    positions_folded = _fold_positions(positions, sc_t1, sc_t2)

    step_suffix = f"_step{total_steps}"
    obs_raw_path = raw_data_dir / f"observables_data{step_suffix}.npz"
    np.savez(obs_raw_path, samples_obs=samples_obs.astype(np.int8), charge_density=charge_density, positions=positions, positions_folded=positions_folded, corr_matrix=corr_matrix, r_values_plot=r_values_plot, g_r_plot=g_r_plot, q_list_raw=q_list_raw, q_list=q_list, s_q=s_q, q_shell_unique=q_shell_unique, s_q_abs=s_q_abs, s_q_abs_err=s_q_abs_err, q_abs=q_abs, s_q_plot=s_q_plot, rel_min=rel_min, g_site=g_site, r_site=r_site, origin_idx=np.array([origin_idx], dtype=np.int64), obs_n_samples=np.array([samples_obs.shape[0]], dtype=np.int64), obs_n_discard_per_chain=np.array([obs_discard_eff], dtype=np.int64), optimization_step=np.array([total_steps], dtype=np.int64), sc_basis=sc_basis, sc_t1=sc_t1, sc_t2=sc_t2)

    rhombus = np.array([np.zeros(2), sc_t1, sc_t1 + sc_t2, sc_t2, np.zeros(2)])
    rhombus_centered = np.array([[-0.5 * sc_t1[0] - 0.5 * sc_t2[0], -0.5 * sc_t1[1] - 0.5 * sc_t2[1]], [0.5 * sc_t1[0] - 0.5 * sc_t2[0], 0.5 * sc_t1[1] - 0.5 * sc_t2[1]], [0.5 * sc_t1[0] + 0.5 * sc_t2[0], 0.5 * sc_t1[1] + 0.5 * sc_t2[1]], [-0.5 * sc_t1[0] + 0.5 * sc_t2[0], -0.5 * sc_t1[1] + 0.5 * sc_t2[1]], [-0.5 * sc_t1[0] - 0.5 * sc_t2[0], -0.5 * sc_t1[1] - 0.5 * sc_t2[1]]], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    scp = ax.scatter(positions_folded[:, 0], positions_folded[:, 1], c=charge_density, s=140, cmap="viridis", vmin=0.0, vmax=float(np.max(charge_density)))
    ax.plot(rhombus[:, 0], rhombus[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title=r"Charge density $\langle n_i \rangle$", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(scp, ax=ax, label=r"$\langle n_i \rangle$")
    _save_fig(fig, plots_obs_dir / f"charge_density{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax.set(title="Pair correlation (minimum-image distances)", xlabel="Distance r (minimum image)", ylabel="g(r)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"pair_correlation_radial{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc2 = ax.scatter(rel_min[:, 0], rel_min[:, 1], c=g_site, s=180, cmap="viridis")
    ax.scatter([0.0], [0.0], s=220, facecolors="none", edgecolors="black", linewidths=1.1)
    ax.plot(rhombus_centered[:, 0], rhombus_centered[:, 1], color="black", lw=1.2, alpha=0.9)
    ax.set(title="Pair correlation mapped to 2D (minimum image)", xlabel="x (minimum image)", ylabel="y (minimum image)", aspect="equal")
    fig.colorbar(sc2, ax=ax, label="g(r)")
    _save_fig(fig, plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(q_shell_unique, s_q_abs, yerr=s_q_abs_err, fmt="o-", ms=4, lw=1.2, capsize=2, color="tab:blue")
    ax.set(title="Static structure factor (radial, WS-BZ folded)", xlabel=r"$|\mathbf{q}|$ (WS-BZ)", ylabel="S(q)")
    ax.grid(alpha=0.25)
    _save_fig(fig, plots_obs_dir / f"static_structure_factor{step_suffix}.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc3 = ax.scatter(q_list[:, 0], q_list[:, 1], c=s_q_plot.real, s=120, cmap="magma", edgecolors="black", linewidths=0.25)
    ax.set(title="Static structure factor in q-space (WS-BZ folded)", xlabel=r"$q_x$", ylabel=r"$q_y$", aspect="equal")
    ax.grid(alpha=0.2)
    fig.colorbar(sc3, ax=ax, label=r"$S(\mathbf{q})$")
    _save_fig(fig, plots_obs_dir / f"static_structure_factor_qspace{step_suffix}.png")
    shutil.copy2(job_dir / "run_script.py", plots_obs_dir / f"run_script_used{step_suffix}.py")

    summary = {
        "job_dir": str(job_dir),
        "backend": backend,
        "model_type": "psiformer",
        "sample_type": sample_type,
        "system": {"supercell_matrix": supercell_matrix.tolist(), "n_supercell_cells": n_sc_cells, "Lx": lx, "Ly": ly, "n_sites": int(graph.n_nodes), "n_fermions": n_fermions, "V1": V1, "t1": t1, "t2": t2, "phi": phi, "m": m},
        "network": {"num_layers": num_layers, "d_model": d_model, "n_heads": n_heads, "mlp_hidden_factor": mlp_hidden_factor, "n_determinants": n_determinants, "use_jastrow": use_jastrow, "embedding_type": embedding_type},
        "optimization": {"n_iter_this_run": n_iter, "start_step": start_step, "end_step_exclusive": end_step, "total_steps": total_steps, "n_samples": n_samples, "n_discard_per_chain": n_discard_per_chain, "n_chains": n_chains, "sweep_size": sweep_size, "optimizer": f"Sgd(learning_rate={learning_rate})", "diag_shift": diag_shift, "mode": "complex"},
        "observables": {"obs_n_samples": int(samples_obs.shape[0]), "obs_n_discard_per_chain": int(obs_discard_eff), "optimization_step": total_steps},
        "n_wavefunction_params": int(nk.jax.tree_size(vstate.parameters)),
        "final_energy_real": float(np.real(energy_mean[-1])) if energy_mean.size > 0 else None,
        "final_energy_sigma": float(energy_sigma[-1]) if energy_sigma.size > 0 else None,
        "final_update_norm": float(un_values[-1]) if un_values.size > 0 else None,
        "ed_reference_energies": {str(n): (None if e is None else float(e)) for n, e in ed_energies.items()},
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
