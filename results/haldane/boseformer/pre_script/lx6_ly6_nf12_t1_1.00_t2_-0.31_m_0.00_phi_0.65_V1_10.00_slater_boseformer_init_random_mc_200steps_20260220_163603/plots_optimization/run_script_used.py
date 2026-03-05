import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
import numpy as np

from aidan_custom.haldane_model import (
    build_haldane_hamiltonian,
    noninteracting_slater_orbitals_haldane,
)
from aidan_custom.models import (
    LogSlaterBoseFormer,
    make_supercell_reciprocal_vectors_from_graph,
)
from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
    static_structure_factor,
)
from aidan_custom.optimization import (
    exact_projected_reference_ground_state_energy,
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
)


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
    history["energy_std_local"] = np.asarray(history["energy_std_local"], dtype=np.float64)
    history["energy_tau"] = np.asarray(history["energy_tau"], dtype=np.float64)
    history["energy_rhat"] = np.asarray(history["energy_rhat"], dtype=np.float64)
    history["update_norm_iters"] = np.asarray(history["update_norm_iters"], dtype=np.int64)
    history["update_norm_values"] = np.asarray(history["update_norm_values"], dtype=np.float64)
    return history


def _concat_1d(old_arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
    # Written with Codex 02-21-26.
    if old_arr.size == 0:
        return new_arr
    if new_arr.size == 0:
        return old_arr
    return np.concatenate((old_arr, new_arr), axis=0)


def main():
    # Written with Codex 02-21-26.
    matplotlib.use("Agg")

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

    # Single-particle parameters from notebooks/haldane_model.ipynb.
    t1 = 1.0
    t2 = -1.0 / (4.0 * np.cos(0.65))
    phi = 0.65
    m = 0.0

    # Many-body system parameters.
    Lx = 6
    Ly = 6
    V1 = 10.0
    n_fermions = 12

    # Optimization controls.
    model_type = "slater_boseformer"
    sample_type = "MC"
    n_iter = int(os.environ.get("N_ITER", "1000"))
    n_samples = 1024 * 4
    n_discard_per_chain = 4
    sweep_size = n_fermions
    n_chains = 512
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.005"))

    # Post-training observables sampling controls.
    obs_n_samples = int(os.environ.get("OBS_N_SAMPLES", str(n_samples * 10)))
    obs_n_discard_per_chain = int(
        os.environ.get("OBS_N_DISCARD_PER_CHAIN", str(max(64, n_discard_per_chain)))
    )

    # BoseFormer controls from notebooks/haldane_model.ipynb.
    boseformer_num_layers = 2
    boseformer_d_model = 32
    boseformer_n_heads = 2
    boseformer_mlp_hidden_factor = 2
    slater_init_mode = "random"

    graph, hi, ham = build_haldane_hamiltonian(
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
        n_fermions=n_fermions,
    )

    assert n_fermions <= graph.n_nodes
    print(f"n_sites={graph.n_nodes}, n_fermions={hi.n_fermions}")
    print(f"hilbert={hi}")
    print(f"V1={V1}")
    print(f"max_conn_size={ham.max_conn_size}")

    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        n_chains=n_chains,
        sweep_size=sweep_size,
    )

    if model_type != "slater_boseformer":
        raise ValueError(f"Unknown model_type={model_type!r}")

    g_vectors = make_supercell_reciprocal_vectors_from_graph(graph)
    positions_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(graph.positions, dtype=np.float64)
    )
    g_vectors_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(g_vectors, dtype=np.float64)
    )

    slater_initial_m_orbitals = None
    if slater_init_mode == "noninteracting":
        slater_initial_m_orbitals = noninteracting_slater_orbitals_haldane(
            graph=graph,
            Lx=Lx,
            Ly=Ly,
            n_fermions=n_fermions,
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
        )
        print("Initializing Slater determinant from non-interacting ground-state orbitals.")
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
        diag_shift=0.01,
        mode="complex",
    )

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=log_optimization_diagnostics)

    (job_dir / "vstate_variables.mpack").write_bytes(
        flax.serialization.to_bytes(vstate.variables)
    )
    param_leaves, param_treedef = jax.tree_util.tree_flatten(vstate.parameters)
    np.savez(
        job_dir / "vstate_parameters_leaves.npz", *[np.asarray(x) for x in param_leaves]
    )
    (job_dir / "vstate_parameters_treedef.txt").write_text(repr(param_treedef) + "\n")

    end_step = start_step + n_iter
    log_prefix = job_dir / f"runtime_log_step{start_step}_to_step{end_step}"
    log.serialize(log_prefix)
    log.serialize(job_dir / "runtime_log")

    energy_history = log.data["Energy"]
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
        update_norm_iters_new = np.asarray(update_norm_history.iters, dtype=np.int64) + start_step
        update_norm_values_new = np.asarray(update_norm_history, dtype=np.float64)
    else:
        update_norm_iters_new = np.asarray([], dtype=np.int64)
        update_norm_values_new = np.asarray([], dtype=np.float64)

    history_all = {
        "iters": _concat_1d(history_prev["iters"], iters_new),
        "energy_mean": _concat_1d(history_prev["energy_mean"], energy_mean_new),
        "energy_sigma": _concat_1d(history_prev["energy_sigma"], energy_sigma_new),
        "energy_variance": _concat_1d(history_prev["energy_variance"], energy_variance_new),
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

    reference_dim_cutoff = 200_000

    hilbert_dim_comb = math.comb(int(hi.n_orbitals), int(hi.n_fermions))
    if hilbert_dim_comb <= reference_dim_cutoff:
        e_ref, e_ref_plot_label, e_ref_method = exact_reference_ground_state_energy(
            hamiltonian=ham,
            Lx=Lx,
            Ly=Ly,
            n_fermions=int(hi.n_fermions),
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
            V1=V1,
        )
    else:
        e_ref = None
        e_ref_plot_label = None
        e_ref_method = (
            f"skipped (C(n_sites,n_fermions)={hilbert_dim_comb} > {reference_dim_cutoff})"
        )

    projected_orbitals = Lx * Ly
    projected_dim_comb = math.comb(int(projected_orbitals), int(hi.n_fermions))
    if projected_dim_comb <= reference_dim_cutoff:
        (
            e_ref_projected,
            e_ref_projected_plot_label,
            e_ref_projected_method,
            e_ref_projected_sector,
        ) = exact_projected_reference_ground_state_energy(
            Lx=Lx,
            Ly=Ly,
            n_fermions=int(hi.n_fermions),
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
            V1=V1,
            selected_bands=(0,),
        )
    else:
        e_ref_projected = None
        e_ref_projected_plot_label = None
        e_ref_projected_method = (
            f"skipped (C(projected_orbitals,n_fermions)={projected_dim_comb} > "
            f"{reference_dim_cutoff})"
        )
        e_ref_projected_sector = None

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
    if e_ref_projected is not None:
        ax.axhline(
            e_ref_projected,
            color="tab:green",
            ls="-.",
            lw=1.5,
            label=e_ref_projected_plot_label,
        )
    if e_ref is not None:
        ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)

    ax.set_title("Haldane model: Slater x BoseFormer optimization")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_opt_dir / "energy_vs_step.png", dpi=180)
    plt.close(fig)

    n_points = iters.size
    start_idx_final = max(int(math.floor(0.9 * n_points)), 0)
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
        if e_ref_projected is not None:
            ax.axhline(
                e_ref_projected,
                color="tab:green",
                ls="-.",
                lw=1.5,
                label=e_ref_projected_plot_label,
            )
        if e_ref is not None:
            ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)
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
    fig.savefig(plots_opt_dir / "energy_vs_step_final10pct.png", dpi=180)
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
        ax.plot(iters[:n_tau], energy_tau[:n_tau], color="tab:orange", lw=1.5, label="TauCorr")
        if energy_rhat.size > 0:
            n_rhat = min(iters.size, energy_rhat.size)
            ax.plot(iters[:n_rhat], energy_rhat[:n_rhat], color="tab:brown", lw=1.5, label="R_hat")
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

    positions = np.asarray(graph.positions, dtype=np.float64)
    basis_vectors = np.asarray(graph.basis_vectors, dtype=np.float64)
    pbc = np.asarray(graph.pbc, dtype=bool)

    pc_data = pair_correlation_cartesian(
        samples=samples_obs,
        positions=positions,
        basis_vectors=basis_vectors,
        pbc=pbc,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
    )
    charge_density = pc_data["charge_density"]
    corr_matrix = pc_data["corr_matrix"]
    dist_matrix = pc_data["dist_matrix"]
    translations = pc_data["translations"]
    r_values_plot = pc_data["r_values_plot"]
    g_r_plot = pc_data["g_r_plot"]

    q_list, s_q = static_structure_factor(
        samples=samples_obs,
        positions=positions,
        basis_vectors=basis_vectors,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
    )

    rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(
        positions=positions,
        basis_coords=np.asarray(graph.basis_coords),
        translations=translations,
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
        corr_matrix=corr_matrix,
        dist_matrix=dist_matrix,
        translations=translations,
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
    )

    fig_charge, ax_charge = plt.subplots(figsize=(5.6, 5.0))
    sc_charge = ax_charge.scatter(
        positions[:, 0],
        positions[:, 1],
        c=charge_density,
        s=220,
        cmap="viridis",
        marker="o",
        vmin=0.0,
        vmax=float(np.max(charge_density)),
    )
    ax_charge.set_title(r"Charge density $\langle n_i \rangle$")
    ax_charge.set_xlabel("x")
    ax_charge.set_ylabel("y")
    ax_charge.set_aspect("equal", "box")
    fig_charge.colorbar(sc_charge, ax=ax_charge, label=r"$\langle n_i \rangle$")
    fig_charge.tight_layout()
    fig_charge.savefig(plots_obs_dir / f"charge_density{step_suffix}.png", dpi=180)
    plt.close(fig_charge)

    fig_pair, ax_pair = plt.subplots(figsize=(6.2, 4.8))
    ax_pair.plot(r_values_plot, g_r_plot, "o-", color="tab:blue", lw=1.4, ms=4)
    ax_pair.set_title("Pair correlation")
    ax_pair.set_xlabel("Cartesian distance r")
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
        s=220,
        cmap="viridis",
        marker="o",
    )
    ax_pair2d.scatter(
        [0.0],
        [0.0],
        s=240,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
    )
    ax_pair2d.set_title("Pair correlation mapped to 2D")
    ax_pair2d.set_xlabel("x (minimum image)")
    ax_pair2d.set_ylabel("y (minimum image)")
    ax_pair2d.set_aspect("equal", "box")
    fig_pair2d.colorbar(sc_pair2d, ax=ax_pair2d, label="g(r)")
    fig_pair2d.tight_layout()
    fig_pair2d.savefig(plots_obs_dir / f"pair_correlation_mapped_2d{step_suffix}.png", dpi=180)
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
    ax_sq.set_title("Static structure factor")
    ax_sq.set_xlabel("q")
    ax_sq.set_ylabel("S(q)")
    ax_sq.grid(alpha=0.25)
    fig_sq.tight_layout()
    fig_sq.savefig(plots_obs_dir / f"static_structure_factor{step_suffix}.png", dpi=180)
    plt.close(fig_sq)

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
            "Lx": Lx,
            "Ly": Ly,
            "n_fermions": n_fermions,
            "V1": V1,
            "t1": t1,
            "t2": float(t2),
            "phi": phi,
            "m": m,
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
            "diag_shift": 0.01,
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
        "reference_method": e_ref_method,
        "reference_energy": None if e_ref is None else float(e_ref),
        "projected_reference_method": e_ref_projected_method,
        "projected_reference_energy": (
            None if e_ref_projected is None else float(e_ref_projected)
        ),
        "projected_reference_sector": (
            None
            if e_ref_projected_sector is None
            else [int(v) for v in e_ref_projected_sector]
        ),
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if e_ref is not None:
        print(f"Exact ground-state reference [{e_ref_method}] = {e_ref:.10f}")
        print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
        print(f"Final error (VMC - reference) = {energy_mean.real[-1] - e_ref:.10f}")
    else:
        print(f"Exact ground-state reference [{e_ref_method}]")
        print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")

    if e_ref_projected is not None:
        print(
            "Projected lowest-band reference "
            f"[{e_ref_projected_method}] = {e_ref_projected:.10f}"
        )
        print(
            "Final error (VMC - projected reference) = "
            f"{energy_mean.real[-1] - e_ref_projected:.10f}"
        )
    else:
        print(f"Projected lowest-band reference [{e_ref_projected_method}]")

    if update_norm_values.size > 0:
        print(f"Final update norm ||dtheta||_2 = {float(update_norm_values[-1]):.6e}")
    print(f"Final local-energy std dev = {float(energy_std_local[-1]):.10f}")
    print(f"Saved observables raw data to: {obs_raw_path}")
    print(f"Saved observables plots in: {plots_obs_dir}")


if __name__ == "__main__":
    main()
