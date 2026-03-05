import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import netket as nk
import numpy as np
from matplotlib import pyplot as plt

from aidan_custom.bloch_ed import build_mote2_three_orbital_lattice_embedding
from aidan_custom.geometry import fold_to_shortest_k
from aidan_custom.models import LogSlaterBoseFormer
from aidan_custom.mote2_three_orbital import (
    MOTE2_A1,
    MOTE2_A2,
    mote2_three_orbital_reciprocal_vectors,
)
from aidan_custom.mote2_three_orbital_model import (
    build_mote2_three_orbital_hamiltonian,
    noninteracting_slater_orbitals_mote2_three_orbital,
)
from aidan_custom.observables import radial_average_structure_factor
from aidan_custom.optimization import make_optimization_callback

matplotlib.use("Agg")


def _empty_history_dict():
    # Written with Codex 02-21-26.
    return {
        "iters": np.asarray([], dtype=np.int64),
        "energy_mean": np.asarray([], dtype=np.complex128),
        "energy_sigma": np.asarray([], dtype=np.float64),
        "energy_variance": np.asarray([], dtype=np.float64),
        "energy_std_local": np.asarray([], dtype=np.float64),
        "energy_tau": np.asarray([], dtype=np.float64),
        "energy_rhat": np.asarray([], dtype=np.float64),
        "update_norm_iters": np.asarray([], dtype=np.int64),
        "update_norm_values": np.asarray([], dtype=np.float64),
    }


def _load_history_dict(path: Path):
    # Written with Codex 02-21-26.
    history = _empty_history_dict()
    if not path.exists():
        return history

    loaded = np.load(path, allow_pickle=False)
    for key in history:
        if key in loaded:
            history[key] = np.asarray(loaded[key])

    history["iters"] = np.asarray(history["iters"], dtype=np.int64)
    history["energy_mean"] = np.asarray(history["energy_mean"], dtype=np.complex128)
    history["energy_sigma"] = np.asarray(history["energy_sigma"], dtype=np.float64)
    history["energy_variance"] = np.asarray(history["energy_variance"], dtype=np.float64)
    history["energy_std_local"] = np.asarray(
        history["energy_std_local"], dtype=np.float64
    )
    history["energy_tau"] = np.asarray(history["energy_tau"], dtype=np.float64)
    history["energy_rhat"] = np.asarray(history["energy_rhat"], dtype=np.float64)
    history["update_norm_iters"] = np.asarray(
        history["update_norm_iters"], dtype=np.int64
    )
    history["update_norm_values"] = np.asarray(
        history["update_norm_values"], dtype=np.float64
    )
    return history


def _concat_1d(old_arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
    # Written with Codex 02-21-26.
    if old_arr.size == 0:
        return new_arr
    if new_arr.size == 0:
        return old_arr
    return np.concatenate((old_arr, new_arr), axis=0)


def _hashable_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    # Written with Codex 02-21-26.
    arr = np.asarray(matrix, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def _mote2_site_positions_and_geometry(
    supercell_matrix: np.ndarray,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    # Written with Codex 02-21-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        supercell_matrix=np.asarray(supercell_matrix, dtype=np.int64),
    )
    n_sites = int(lattice.Lx) * int(lattice.Ly) * int(lattice.n_orbitals_per_cell)

    a1 = float(a_m) * np.asarray(MOTE2_A1, dtype=np.float64)
    a2 = float(a_m) * np.asarray(MOTE2_A2, dtype=np.float64)
    basis_vectors = np.stack([a1, a2], axis=0)
    u0 = np.asarray([float(a_m) / np.sqrt(3.0), 0.0], dtype=np.float64)
    orbital_offsets = np.asarray([u0, np.zeros(2), -u0], dtype=np.float64)

    positions = np.empty((n_sites, 2), dtype=np.float64)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        cell_bravais = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = (
            float(cell_bravais[0]) * a1
            + float(cell_bravais[1]) * a2
            + orbital_offsets[int(orb)]
        )

    supercell_t1 = (
        float(lattice.supercell_matrix[0, 0]) * a1
        + float(lattice.supercell_matrix[1, 0]) * a2
    )
    supercell_t2 = (
        float(lattice.supercell_matrix[0, 1]) * a1
        + float(lattice.supercell_matrix[1, 1]) * a2
    )
    return positions, basis_vectors, supercell_t1, supercell_t2, lattice


def _supercell_reciprocal_vectors(
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    supercell_vectors = np.stack(
        (
            np.asarray(supercell_t1, dtype=np.float64),
            np.asarray(supercell_t2, dtype=np.float64),
        ),
        axis=0,
    )
    return np.asarray(
        (2.0 * np.pi) * np.linalg.inv(supercell_vectors.T),
        dtype=np.float64,
    )


def _supercell_matrix_columns(
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    return np.column_stack(
        (
            np.asarray(supercell_t1, dtype=np.float64),
            np.asarray(supercell_t2, dtype=np.float64),
        )
    )


def _minimum_image_vector(
    delta: np.ndarray,
    supercell_cols: np.ndarray,
    search_radius: int = 2,
) -> tuple[np.ndarray, float]:
    # Written with Codex 02-21-26.
    frac = np.linalg.solve(supercell_cols, np.asarray(delta, dtype=np.float64))
    base = np.rint(frac).astype(np.int64)

    best_vec = np.zeros(2, dtype=np.float64)
    best_norm2 = np.inf
    for dn1 in range(-int(search_radius), int(search_radius) + 1):
        for dn2 in range(-int(search_radius), int(search_radius) + 1):
            shift = base + np.asarray((dn1, dn2), dtype=np.int64)
            vec = np.asarray(delta, dtype=np.float64) - supercell_cols @ shift
            norm2 = float(np.dot(vec, vec))
            if norm2 < best_norm2:
                best_norm2 = norm2
                best_vec = vec
    return best_vec, float(np.sqrt(best_norm2))


def _fold_positions_to_rhombus(
    positions: np.ndarray,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    supercell_cols = _supercell_matrix_columns(
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )
    frac = np.linalg.solve(supercell_cols, np.asarray(positions, dtype=np.float64).T).T
    frac_folded = frac - np.floor(frac)
    pos_folded = (supercell_cols @ frac_folded.T).T
    return np.asarray(pos_folded, dtype=np.float64), np.asarray(frac_folded, dtype=np.float64)


def _pair_correlation_from_samples(
    samples: np.ndarray,
    positions: np.ndarray,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
    n_fermions: int,
) -> dict[str, np.ndarray]:
    # Written with Codex 02-21-26.
    n_samples = int(samples.shape[0])
    n_sites = int(samples.shape[1])

    charge_density = samples.mean(axis=0)
    corr_matrix = (samples.T @ samples) / float(n_samples)

    supercell_cols = _supercell_matrix_columns(
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )
    dist_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            _, d_ij = _minimum_image_vector(
                delta=np.asarray(positions[j] - positions[i], dtype=np.float64),
                supercell_cols=supercell_cols,
                search_radius=2,
            )
            dist_matrix[i, j] = d_ij
            dist_matrix[j, i] = d_ij

    offdiag = ~np.eye(n_sites, dtype=bool)
    dist_shell = np.round(dist_matrix, decimals=12)
    r_values = np.unique(dist_shell[offdiag])

    norm_density = (float(n_fermions) / float(n_sites)) ** 2
    g_r = np.empty(r_values.size, dtype=np.float64)
    for idx, r_shell in enumerate(r_values):
        mask = (dist_shell == r_shell) & offdiag
        if np.any(mask):
            g_r[idx] = float(np.real(corr_matrix[mask].mean())) / norm_density
        else:
            g_r[idx] = np.nan

    if r_values.size == 0 or not np.isclose(r_values[0], 0.0):
        r_values_plot = np.concatenate((np.asarray([0.0], dtype=np.float64), r_values))
        g_r_plot = np.concatenate((np.asarray([0.0], dtype=np.float64), g_r))
    else:
        r_values_plot = r_values.copy()
        g_r_plot = g_r.copy()
        g_r_plot[0] = 0.0

    return {
        "charge_density": charge_density,
        "corr_matrix": corr_matrix,
        "dist_matrix": dist_matrix,
        "r_values_plot": r_values_plot,
        "g_r_plot": g_r_plot,
    }


def _map_radial_g_to_minimum_image_rhombus(
    positions: np.ndarray,
    basis_coords: np.ndarray,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
    r_values_plot: np.ndarray,
    g_r_plot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # Written with Codex 02-21-26.
    n_sites = int(positions.shape[0])
    coords = np.asarray(basis_coords, dtype=np.int64)

    a_sublattice = np.where(coords[:, 2] == 0)[0]
    a_origin = np.where((coords[:, 0] == 0) & (coords[:, 1] == 0) & (coords[:, 2] == 0))[0]
    if a_origin.size > 0:
        origin_idx = int(a_origin[0])
    elif a_sublattice.size > 0:
        origin_idx = int(a_sublattice[0])
    else:
        raise ValueError("Could not find an A-sublattice site for the g(r) origin.")

    supercell_cols = _supercell_matrix_columns(
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )
    origin_pos = np.asarray(positions[origin_idx], dtype=np.float64)
    rel_min = np.empty((n_sites, 2), dtype=np.float64)
    r_site = np.empty(n_sites, dtype=np.float64)
    for i in range(n_sites):
        best_vec, best_dist = _minimum_image_vector(
            delta=np.asarray(positions[i], dtype=np.float64) - origin_pos,
            supercell_cols=supercell_cols,
            search_radius=2,
        )
        rel_min[i] = best_vec
        r_site[i] = best_dist

    shell_to_g = {
        float(np.round(r, 12)): float(g)
        for r, g in zip(np.asarray(r_values_plot, dtype=np.float64), np.asarray(g_r_plot, dtype=np.float64))
    }
    r_shell = np.round(r_site, 12)
    g_site = np.empty(n_sites, dtype=np.float64)
    for idx, rs in enumerate(r_shell):
        key = float(rs)
        if key in shell_to_g:
            g_site[idx] = shell_to_g[key]
        else:
            nearest = int(np.argmin(np.abs(np.asarray(r_values_plot, dtype=np.float64) - float(r_site[idx]))))
            g_site[idx] = float(np.asarray(g_r_plot, dtype=np.float64)[nearest])
    return rel_min, g_site, origin_idx, r_site


def _q_vectors_from_lattice(
    lattice,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))

    q_vectors_raw = np.empty((int(lattice.Lx) * int(lattice.Ly), 2), dtype=np.float64)
    idx = 0
    for kx in range(int(lattice.Lx)):
        for ky in range(int(lattice.Ly)):
            coeff = lattice.kpoint_coefficients[kx, ky]
            q_raw = coeff[0] * b1 + coeff[1] * b2
            q_vectors_raw[idx] = q_raw
            idx += 1

    n_q = int(q_vectors_raw.shape[0])
    q_vectors_ws = np.empty_like(q_vectors_raw)
    q_vectors_ws_base = np.empty_like(q_vectors_raw)
    for i in range(n_q):
        q_vectors_ws_base[i] = fold_to_shortest_k(
            kvec=np.asarray(q_vectors_raw[i], dtype=np.float64),
            b1=np.asarray(b1, dtype=np.float64),
            b2=np.asarray(b2, dtype=np.float64),
            search_radius=8,
        )

    inv_partner = np.empty(n_q, dtype=np.int64)
    for i in range(n_q):
        best_j = -1
        best_norm = np.inf
        for j in range(n_q):
            resid = fold_to_shortest_k(
                kvec=np.asarray(q_vectors_raw[i] + q_vectors_raw[j], dtype=np.float64),
                b1=np.asarray(b1, dtype=np.float64),
                b2=np.asarray(b2, dtype=np.float64),
                search_radius=8,
            )
            resid_norm = float(np.linalg.norm(resid))
            if resid_norm < best_norm:
                best_norm = resid_norm
                best_j = j
        if best_j < 0 or best_norm > 1.0e-9:
            raise RuntimeError(
                f"Failed to find inverse momentum partner for index {i} "
                f"(best residual={best_norm:.3e})."
            )
        inv_partner[i] = int(best_j)

    assigned = np.zeros(n_q, dtype=bool)
    for i in range(n_q):
        if assigned[i]:
            continue
        j = int(inv_partner[i])
        qi = np.asarray(q_vectors_ws_base[i], dtype=np.float64)
        if i == j:
            q_vectors_ws[i] = qi
            assigned[i] = True
            continue

        q_vectors_ws[i] = qi
        q_vectors_ws[j] = -qi
        assigned[i] = True
        assigned[j] = True

    return q_vectors_raw, q_vectors_ws


def _structure_factor_from_corr_matrix(
    corr_matrix: np.ndarray,
    positions: np.ndarray,
    q_vectors: np.ndarray,
    n_fermions: int,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    phase = np.exp(-1j * np.tensordot(disp, q_vectors, axes=([2], [1])))
    s_q = np.einsum("ij,ijq->q", corr_matrix, phase, optimize=True) / float(n_fermions)
    return np.real(s_q)


def main():
    # Written with Codex 02-21-26.
    devices = jax.devices()
    backend = jax.default_backend()
    print("devices:", devices)
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend}. Refusing to run on CPU."
        )

    job_dir = Path(__file__).resolve().parent
    raw_data_dir = job_dir / "raw_data"
    plots_opt_dir = job_dir / "plots_optimization"
    plots_obs_dir = job_dir / "plots_observables"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_opt_dir.mkdir(parents=True, exist_ok=True)
    plots_obs_dir.mkdir(parents=True, exist_ok=True)

    history_path = raw_data_dir / "optimization_history.npz"
    checkpoint_path = job_dir / "vstate_variables.mpack"

    # Ideal-band parameters from notebooks/mote2_netket.ipynb.
    delta = 5.426542963374432
    ez = 0.0
    t_th1 = 1.0
    t_hh1 = 0.4689334070165771
    t_th2 = -0.07151471515876148
    t_hh3 = 0.037340187144172164
    t_tt1 = -0.0377535889269079
    a_m = 1.0
    ph_conj = True

    # Many-body system parameters.
    supercell_matrix = np.array([[3, 6], [-3, 3]], dtype=np.int64)
    n_supercell_cells = int(
        abs(
            supercell_matrix[0, 0] * supercell_matrix[1, 1]
            - supercell_matrix[0, 1] * supercell_matrix[1, 0]
        )
    )
    if n_supercell_cells != 27:
        raise RuntimeError(
            "Expected the requested 27-cell geometry, got "
            f"det(supercell_matrix)={n_supercell_cells}."
        )
    V1 = 0.3
    n_fermions = 9

    # Optimization controls.
    model_type = "slater_boseformer"
    sample_type = os.environ.get("SAMPLE_TYPE", "MC")
    n_iter = int(os.environ.get("N_ITER", "2000"))
    n_samples = int(os.environ.get("N_SAMPLES", str(1024 * 4)))
    n_discard_per_chain = int(os.environ.get("N_DISCARD_PER_CHAIN", "4"))
    sweep_size = int(os.environ.get("SWEEP_SIZE", str(2 * n_fermions)))
    n_chains = int(os.environ.get("N_CHAINS", "1024"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.05"))
    diag_shift = float(os.environ.get("DIAG_SHIFT", "0.001"))

    # Post-training observables sampling controls.
    obs_n_samples = int(os.environ.get("OBS_N_SAMPLES", 1_000_000))
    obs_n_discard_per_chain = int(
        os.environ.get(
            "OBS_N_DISCARD_PER_CHAIN",
            str(max(64, n_discard_per_chain)),
        )
    )

    # BoseFormer controls.
    boseformer_num_layers = int(os.environ.get("BOSEFORMER_NUM_LAYERS", "4"))
    boseformer_d_model = int(os.environ.get("BOSEFORMER_D_MODEL", "32"))
    boseformer_n_heads = int(os.environ.get("BOSEFORMER_N_HEADS", "4"))
    boseformer_mlp_hidden_factor = int(
        os.environ.get("BOSEFORMER_MLP_HIDDEN_FACTOR", "4")
    )
    slater_init_mode = os.environ.get("SLATER_INIT_MODE", "random")

    graph, hi, ham = build_mote2_three_orbital_hamiltonian(
        supercell_matrix=supercell_matrix,
        n_fermions=n_fermions,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
        V1=V1,
    )

    assert n_fermions <= graph.n_nodes
    print(f"n_sites={graph.n_nodes}, n_fermions={hi.n_fermions}")
    print(f"supercell_matrix={supercell_matrix.tolist()}")
    print(f"n_supercell_cells={n_supercell_cells}")
    print(f"hilbert={hi}")
    print(f"V1={V1}")
    print(f"max_conn_size={ham.max_conn_size}")
    if int(graph.n_nodes) != 3 * n_supercell_cells:
        raise RuntimeError(
            "Unexpected site count for MoTe2 three-orbital supercell: "
            f"n_sites={graph.n_nodes}, expected={3 * n_supercell_cells}."
        )

    if model_type != "slater_boseformer":
        raise ValueError(f"Unknown model_type={model_type!r}")

    (
        positions,
        basis_vectors,
        supercell_t1,
        supercell_t2,
        lattice,
    ) = _mote2_site_positions_and_geometry(
        supercell_matrix=supercell_matrix,
        a_m=a_m,
    )
    g_vectors = _supercell_reciprocal_vectors(
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )
    positions_hashable = _hashable_matrix(positions)
    g_vectors_hashable = _hashable_matrix(g_vectors)

    slater_initial_m_orbitals = None
    if slater_init_mode == "noninteracting":
        slater_initial_m_orbitals = noninteracting_slater_orbitals_mote2_three_orbital(
            supercell_matrix=supercell_matrix,
            n_fermions=n_fermions,
            delta=delta,
            ez=ez,
            t_th1=t_th1,
            t_hh1=t_hh1,
            t_th2=t_th2,
            t_hh3=t_hh3,
            t_tt1=t_tt1,
            a_m=a_m,
            ph_conj=ph_conj,
        )
        print("Initializing Slater determinant from non-interacting orbitals.")
    elif slater_init_mode != "random":
        raise ValueError(f"Unknown slater_init_mode={slater_init_mode!r}")

    model = LogSlaterBoseFormer(
        hilbert=hi,
        positions=positions_hashable,
        g_vectors=g_vectors_hashable,
        num_layers=boseformer_num_layers,
        d_model=boseformer_d_model,
        n_heads=boseformer_n_heads,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=slater_initial_m_orbitals,
        boseformer_param_dtype=jnp.float64,
        mlp_hidden_factor=boseformer_mlp_hidden_factor,
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
        )
        ham_sr = ham
    elif sample_type == "FullSum":
        if isinstance(ham, nk.operator.FermionOperator2nd):
            ham_sr = ham
        else:
            ham_sr = ham.to_fermionoperator2nd()
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        raise ValueError(f"Unknown sample_type={sample_type!r}")

    n_wavefunction_params = nk.jax.tree_size(vstate.parameters)
    print(f"total # of wavefunction parameters: {n_wavefunction_params}")

    history_prev = _load_history_dict(history_path)
    if history_prev["iters"].size > 0:
        start_step = int(history_prev["iters"][-1]) + 1
        print(
            f"Detected existing optimization history with {history_prev['iters'].size} points; "
            f"continuing from step {start_step}."
        )
    else:
        start_step = 0
        print("No prior optimization history found; starting from step 0.")

    if checkpoint_path.exists():
        vstate.variables = flax.serialization.from_bytes(
            vstate.variables, checkpoint_path.read_bytes()
        )
        print(f"Loaded checkpoint variables from: {checkpoint_path}")
    elif start_step > 0:
        raise FileNotFoundError(
            f"Expected checkpoint for continuation at {checkpoint_path}, but none was found."
        )

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        mode="complex",
    )

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=make_optimization_callback(job_dir))

    checkpoint_path.write_bytes(flax.serialization.to_bytes(vstate.variables))
    param_leaves, param_treedef = jax.tree_util.tree_flatten(vstate.parameters)
    np.savez(
        job_dir / "vstate_parameters_leaves.npz",
        *[np.asarray(x) for x in param_leaves],
    )
    (job_dir / "vstate_parameters_treedef.txt").write_text(repr(param_treedef) + "\n")

    end_step = start_step + n_iter
    log_prefix = job_dir / f"runtime_log_step{start_step}_to_step{end_step}"
    log.serialize(log_prefix)
    log.serialize(job_dir / "runtime_log")

    energy_history = log.data.get("Energy", None)
    if energy_history is None:
        if history_prev["iters"].size == 0:
            raise RuntimeError(
                "No optimization history is available and n_iter produced no Energy logs. "
                "Increase N_ITER to run at least one optimization step."
            )
        iters_new = np.asarray([], dtype=np.int64)
        energy_mean_new = np.asarray([], dtype=np.complex128)
        energy_sigma_new = np.asarray([], dtype=np.float64)
        energy_variance_new = np.asarray([], dtype=np.float64)
        energy_tau_new = np.asarray([], dtype=np.float64)
        energy_rhat_new = np.asarray([], dtype=np.float64)
    else:
        iters_new = np.asarray(energy_history.iters, dtype=np.int64) + start_step
        energy_mean_new = np.asarray(energy_history["Mean"], dtype=np.complex128)
        energy_sigma_new = np.asarray(energy_history["Sigma"], dtype=np.float64)
        energy_variance_new = np.asarray(energy_history["Variance"], dtype=np.float64)

        try:
            energy_tau_new = np.asarray(energy_history["TauCorr"], dtype=np.float64)
        except Exception:
            energy_tau_new = np.asarray([], dtype=np.float64)

        try:
            energy_rhat_new = np.asarray(energy_history["R_hat"], dtype=np.float64)
        except Exception:
            energy_rhat_new = np.asarray([], dtype=np.float64)

    update_norm_history = log.data.get("UpdateNormL2", None)
    if update_norm_history is not None:
        update_norm_iters_new = (
            np.asarray(update_norm_history.iters, dtype=np.int64) + start_step
        )
        update_norm_values_new = np.asarray(update_norm_history, dtype=np.float64)
    else:
        update_norm_iters_new = np.asarray([], dtype=np.int64)
        update_norm_values_new = np.asarray([], dtype=np.float64)

    history_all = {
        "iters": _concat_1d(history_prev["iters"], iters_new),
        "energy_mean": _concat_1d(history_prev["energy_mean"], energy_mean_new),
        "energy_sigma": _concat_1d(history_prev["energy_sigma"], energy_sigma_new),
        "energy_variance": _concat_1d(
            history_prev["energy_variance"], energy_variance_new
        ),
        "energy_tau": _concat_1d(history_prev["energy_tau"], energy_tau_new),
        "energy_rhat": _concat_1d(history_prev["energy_rhat"], energy_rhat_new),
        "update_norm_iters": _concat_1d(
            history_prev["update_norm_iters"], update_norm_iters_new
        ),
        "update_norm_values": _concat_1d(
            history_prev["update_norm_values"], update_norm_values_new
        ),
    }
    history_all["energy_std_local"] = np.sqrt(
        np.maximum(np.real(history_all["energy_variance"]), 0.0)
    )

    np.savez(
        history_path,
        iters=history_all["iters"],
        energy_mean=history_all["energy_mean"],
        energy_sigma=history_all["energy_sigma"],
        energy_variance=history_all["energy_variance"],
        energy_std_local=history_all["energy_std_local"],
        energy_tau=history_all["energy_tau"],
        energy_rhat=history_all["energy_rhat"],
        update_norm_iters=history_all["update_norm_iters"],
        update_norm_values=history_all["update_norm_values"],
    )

    iters = history_all["iters"]
    energy_mean = history_all["energy_mean"]
    energy_sigma = history_all["energy_sigma"]
    energy_std_local = history_all["energy_std_local"]
    energy_tau = history_all["energy_tau"]
    energy_rhat = history_all["energy_rhat"]
    update_norm_iters = history_all["update_norm_iters"]
    update_norm_values = history_all["update_norm_values"]
    total_steps = int(iters[-1]) + 1 if iters.size > 0 else start_step

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.real(energy_mean), lw=1.8, color="tab:blue", label="VMC_SR")
    ax.fill_between(
        iters,
        np.real(energy_mean) - energy_sigma,
        np.real(energy_mean) + energy_sigma,
        color="tab:blue",
        alpha=0.2,
        linewidth=0,
    )
    ax.set_title("MoTe2 three-orbital: Slater x BoseFormer optimization")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "energy_vs_step.png", dpi=180)
    plt.close(fig)

    n_points = iters.size
    start_idx_final = max(int(math.floor(0.2 * n_points)), 0)
    if start_idx_final >= n_points and n_points > 0:
        start_idx_final = n_points - 1
    iters_final = iters[start_idx_final:]
    energy_mean_final = energy_mean[start_idx_final:]
    energy_sigma_final = energy_sigma[start_idx_final:]

    fig, ax = plt.subplots(figsize=(7, 4))
    if iters_final.size > 0:
        ax.plot(
            iters_final,
            np.real(energy_mean_final),
            lw=1.8,
            color="tab:blue",
            label="VMC_SR",
        )
        ax.fill_between(
            iters_final,
            np.real(energy_mean_final) - energy_sigma_final,
            np.real(energy_mean_final) + energy_sigma_final,
            color="tab:blue",
            alpha=0.2,
            linewidth=0,
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No energy data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "energy_vs_step_final80pct.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        iters,
        np.log10(np.maximum(energy_std_local, 1e-30)),
        color="tab:purple",
        lw=1.5,
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("log10 Std(E_loc)")
    ax.set_title("Local energy standard deviation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "local_energy_std_log10.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if update_norm_values.size > 0:
        ax.plot(update_norm_iters, update_norm_values, color="tab:red", lw=1.5)
        ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "Update norm not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"||$\Delta\theta$||$_2$")
    ax.set_title("Update norm")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "update_norm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if energy_tau.size > 0:
        n_tau = min(iters.size, energy_tau.size)
        ax.plot(
            iters[:n_tau],
            energy_tau[:n_tau],
            color="tab:orange",
            lw=1.5,
            label="TauCorr",
        )
        if energy_rhat.size > 0:
            n_rhat = min(iters.size, energy_rhat.size)
            ax.plot(
                iters[:n_rhat],
                energy_rhat[:n_rhat],
                color="tab:brown",
                lw=1.5,
                label="R_hat",
            )
        ax.legend()
        ax.set_title("Sampling diagnostics")
    else:
        ax.text(
            0.5,
            0.5,
            "No extra diagnostics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Sampling diagnostics")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "sampling_diagnostics.png", dpi=180)
    plt.close(fig)

    print(
        f"Post-training observables sampling at step {total_steps}: "
        f"n_samples={obs_n_samples}, n_discard_per_chain={obs_n_discard_per_chain}"
    )
    samples_obs = np.asarray(
        vstate.sample(
            n_samples=obs_n_samples,
            n_discard_per_chain=obs_n_discard_per_chain,
        )
    ).reshape(-1, hi.size)
    samples_obs = samples_obs.astype(np.float64, copy=False)

    positions_folded, positions_frac_folded = _fold_positions_to_rhombus(
        positions=positions,
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )
    pc_data = _pair_correlation_from_samples(
        samples=samples_obs,
        positions=positions,
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
        n_fermions=n_fermions,
    )
    charge_density = pc_data["charge_density"]
    corr_matrix = pc_data["corr_matrix"]
    dist_matrix = pc_data["dist_matrix"]
    r_values_plot = pc_data["r_values_plot"]
    g_r_plot = pc_data["g_r_plot"]

    q_list_raw, q_list = _q_vectors_from_lattice(
        lattice=lattice,
        a_m=a_m,
    )
    s_q = _structure_factor_from_corr_matrix(
        corr_matrix=corr_matrix,
        positions=positions,
        q_vectors=q_list,
        n_fermions=n_fermions,
    )

    rel_min, g_site, origin_idx, r_site = _map_radial_g_to_minimum_image_rhombus(
        positions=positions,
        basis_coords=np.asarray(lattice.site_to_cell),
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
    )
    q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot = radial_average_structure_factor(
        q_list,
        s_q,
        set_q0_to_zero=True,
    )

    step_suffix = f"_step{total_steps}"
    obs_raw_path = raw_data_dir / f"observables_data{step_suffix}.npz"
    np.savez(
        obs_raw_path,
        samples_obs=samples_obs.astype(np.int8),
        charge_density=charge_density,
        positions=positions,
        positions_folded=positions_folded,
        positions_frac_folded=positions_frac_folded,
        q_list_raw=q_list_raw,
        corr_matrix=corr_matrix,
        dist_matrix=dist_matrix,
        r_values_plot=r_values_plot,
        g_r_plot=g_r_plot,
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
        origin_idx=np.asarray([origin_idx], dtype=np.int64),
        obs_n_samples=np.asarray([samples_obs.shape[0]], dtype=np.int64),
        obs_n_discard_per_chain=np.asarray([obs_n_discard_per_chain], dtype=np.int64),
        optimization_step=np.asarray([total_steps], dtype=np.int64),
        basis_vectors=basis_vectors,
        supercell_t1=supercell_t1,
        supercell_t2=supercell_t2,
    )

    fig_charge, ax_charge = plt.subplots(figsize=(5.8, 5.0))
    sc_charge = ax_charge.scatter(
        positions_folded[:, 0],
        positions_folded[:, 1],
        c=charge_density,
        s=140,
        cmap="viridis",
        marker="o",
        vmin=0.0,
        vmax=float(np.max(charge_density)),
    )
    ax_charge.set_title(r"Charge density $\langle n_i \rangle$")
    ax_charge.set_xlabel("x")
    ax_charge.set_ylabel("y")
    rhombus = np.asarray(
        [
            np.zeros(2, dtype=np.float64),
            np.asarray(supercell_t1, dtype=np.float64),
            np.asarray(supercell_t1, dtype=np.float64) + np.asarray(supercell_t2, dtype=np.float64),
            np.asarray(supercell_t2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ],
        dtype=np.float64,
    )
    ax_charge.plot(rhombus[:, 0], rhombus[:, 1], color="black", lw=1.2, alpha=0.9)
    ax_charge.set_aspect("equal", "box")
    fig_charge.colorbar(sc_charge, ax=ax_charge, label=r"$\langle n_i \rangle$")
    fig_charge.tight_layout()
    fig_charge.savefig(plots_obs_dir / f"charge_density{step_suffix}.png", dpi=180)
    plt.close(fig_charge)

    fig_pair, ax_pair = plt.subplots(figsize=(6.2, 4.8))
    ax_pair.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax_pair.set_title("Pair correlation")
    ax_pair.set_xlabel("Distance r")
    ax_pair.set_ylabel("g(r)")
    ax_pair.grid(alpha=0.25)
    fig_pair.tight_layout()
    fig_pair.savefig(plots_obs_dir / f"pair_correlation_radial{step_suffix}.png", dpi=180)
    plt.close(fig_pair)

    fig_pair2d, ax_pair2d = plt.subplots(figsize=(5.8, 5.2))
    sc_pair2d = ax_pair2d.scatter(
        rel_min[:, 0],
        rel_min[:, 1],
        c=g_site,
        s=180,
        cmap="viridis",
        marker="o",
    )
    ax_pair2d.scatter(
        [0.0],
        [0.0],
        s=220,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
    )
    ax_pair2d.set_title("Pair correlation mapped to 2D")
    ax_pair2d.set_xlabel("x (minimum image)")
    ax_pair2d.set_ylabel("y (minimum image)")
    rhombus_centered = np.asarray(
        [
            -0.5 * np.asarray(supercell_t1, dtype=np.float64) - 0.5 * np.asarray(supercell_t2, dtype=np.float64),
            0.5 * np.asarray(supercell_t1, dtype=np.float64) - 0.5 * np.asarray(supercell_t2, dtype=np.float64),
            0.5 * np.asarray(supercell_t1, dtype=np.float64) + 0.5 * np.asarray(supercell_t2, dtype=np.float64),
            -0.5 * np.asarray(supercell_t1, dtype=np.float64) + 0.5 * np.asarray(supercell_t2, dtype=np.float64),
            -0.5 * np.asarray(supercell_t1, dtype=np.float64) - 0.5 * np.asarray(supercell_t2, dtype=np.float64),
        ],
        dtype=np.float64,
    )
    ax_pair2d.plot(
        rhombus_centered[:, 0],
        rhombus_centered[:, 1],
        color="black",
        lw=1.2,
        alpha=0.9,
    )
    ax_pair2d.set_aspect("equal", "box")
    fig_pair2d.colorbar(sc_pair2d, ax=ax_pair2d, label="g(r)")
    fig_pair2d.tight_layout()
    fig_pair2d.savefig(
        plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png",
        dpi=180,
    )
    plt.close(fig_pair2d)

    fig_sq, ax_sq = plt.subplots(figsize=(6.2, 4.0))
    ax_sq.errorbar(
        q_shell_unique,
        s_q_abs,
        yerr=s_q_abs_err,
        fmt="o-",
        ms=4,
        lw=1.2,
        capsize=2,
        color="tab:blue",
    )
    ax_sq.set_title("Static structure factor (radial)")
    ax_sq.set_xlabel(r"$|\mathbf{q}|$")
    ax_sq.set_ylabel("S(q)")
    ax_sq.grid(alpha=0.25)
    fig_sq.tight_layout()
    fig_sq.savefig(plots_obs_dir / f"static_structure_factor{step_suffix}.png", dpi=180)
    plt.close(fig_sq)

    fig_sq_map, ax_sq_map = plt.subplots(figsize=(5.8, 5.2))
    sc_sq_map = ax_sq_map.scatter(
        q_list[:, 0],
        q_list[:, 1],
        c=s_q_plot.real,
        s=120,
        cmap="magma",
        marker="o",
        edgecolors="black",
        linewidths=0.25,
    )
    ax_sq_map.set_title("Static structure factor in q-space")
    ax_sq_map.set_xlabel(r"$q_x$")
    ax_sq_map.set_ylabel(r"$q_y$")
    ax_sq_map.set_aspect("equal", "box")
    ax_sq_map.grid(alpha=0.2)
    fig_sq_map.colorbar(sc_sq_map, ax=ax_sq_map, label=r"$S(\mathbf{q})$")
    fig_sq_map.tight_layout()
    fig_sq_map.savefig(
        plots_obs_dir / f"static_structure_factor_qspace{step_suffix}.png",
        dpi=180,
    )
    plt.close(fig_sq_map)

    script_text = Path(__file__).read_text(encoding="utf-8")
    (plots_opt_dir / "run_script_used.py").write_text(script_text, encoding="utf-8")
    (plots_obs_dir / f"run_script_used{step_suffix}.py").write_text(
        script_text,
        encoding="utf-8",
    )

    summary = {
        "job_dir": str(job_dir),
        "backend": backend,
        "model_type": model_type,
        "sample_type": sample_type,
        "system": {
            "Lx": int(lattice.Lx),
            "Ly": int(lattice.Ly),
            "n_orbitals_per_cell": int(lattice.n_orbitals_per_cell),
            "n_supercell_cells": int(n_supercell_cells),
            "supercell_matrix": np.asarray(supercell_matrix, dtype=np.int64).tolist(),
            "n_sites": int(graph.n_nodes),
            "n_fermions": n_fermions,
            "V1": V1,
            "delta": delta,
            "ez": ez,
            "t_th1": t_th1,
            "t_hh1": t_hh1,
            "t_th2": t_th2,
            "t_hh3": t_hh3,
            "t_tt1": t_tt1,
            "a_m": a_m,
            "ph_conj": ph_conj,
        },
        "network": {
            "num_layers": boseformer_num_layers,
            "d_model": boseformer_d_model,
            "n_heads": boseformer_n_heads,
            "mlp_hidden_factor": boseformer_mlp_hidden_factor,
            "slater_init_mode": slater_init_mode,
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
            "history_path": str(history_path),
            "runtime_log_this_run": str(log_prefix) + ".json",
        },
        "observables": {
            "obs_n_samples": int(samples_obs.shape[0]),
            "obs_n_discard_per_chain": obs_n_discard_per_chain,
            "optimization_step": total_steps,
            "saved_raw_data": str(obs_raw_path),
            "saved_plots_dir": str(plots_obs_dir),
        },
        "n_wavefunction_params": int(n_wavefunction_params),
        "final_energy_real": float(np.real(energy_mean[-1])) if energy_mean.size > 0 else None,
        "final_energy_sigma": float(energy_sigma[-1]) if energy_sigma.size > 0 else None,
        "final_update_norm": float(update_norm_values[-1]) if update_norm_values.size > 0 else None,
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
    if update_norm_values.size > 0:
        print(f"Final update norm ||dtheta||_2 = {float(update_norm_values[-1]):.6e}")
    print(f"Final local-energy std dev = {float(energy_std_local[-1]):.10f}")
    print(f"Saved observables raw data to: {obs_raw_path}")
    print(f"Saved observables plots in: {plots_obs_dir}")


if __name__ == "__main__":
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

    _log_path = Path(__file__).resolve().parent / "run.log"
    _log_file = open(_log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"\n{'=' * 60}\nRun started: {datetime.now().isoformat()}\n{'=' * 60}\n")
    main()
