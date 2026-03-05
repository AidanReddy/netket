import os
import json
import math
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax

from aidan_custom.haldane_model import (
    build_haldane_hamiltonian,
    noninteracting_slater_orbitals_haldane,
)
from aidan_custom.models import (
    LogSlaterBoseFormer,
    make_supercell_reciprocal_vectors_from_graph,
)
from aidan_custom.optimization import (
    exact_projected_reference_ground_state_energy,
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
)


def main():
    # Written with Codex 02-20-26.
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
    plots_dir = job_dir / "plots_optimization"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Single-particle parameters from notebooks/haldane_model.ipynb.
    t1 = 1.0
    t2 = -1.0 / (4.0 * np.cos(0.65))
    phi = 0.65
    m = 0.0

    # Many-body system parameters from notebooks/haldane_model.ipynb.
    Lx = 5
    Ly = 3
    V1 = 10.0
    n_fermions = 5

    # Optimization controls from notebooks/haldane_model.ipynb.
    model_type = "slater_boseformer"
    sample_type = "MC"
    n_iter = 1_000
    n_samples = 1024 * 16
    mc_chunk_size = 1024
    n_discard_per_chain = 4
    sweep_size = n_fermions
    n_chains = 512
    resume_learning_rate = 0.005
    resume_source_job_dir = (
        Path(__file__).resolve().parent.parent
        / "lx5_ly3_nf5_t1_1.00_t2_-0.31_m_0.00_phi_0.65_V1_10.00_"
        "slater_boseformer_init_random_mc_resume1000_from500_20260220_123933"
    )
    resume_variables_path = Path(
        os.environ.get(
            "RESUME_VSTATE_PATH",
            str(resume_source_job_dir / "vstate_variables.mpack"),
        )
    )
    postprocess_only = os.environ.get("POSTPROCESS_ONLY", "0") == "1"
    if postprocess_only:
        print(
            "POSTPROCESS_ONLY=1: loading existing optimization history and "
            "skipping optimization."
        )
    else:
        print(f"Resuming from checkpoint variables at: {resume_variables_path}")

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
            chunk_size=mc_chunk_size,
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

    if postprocess_only:
        history_path = raw_data_dir / "optimization_history.npz"
        if not history_path.exists():
            raise FileNotFoundError(
                f"Missing {history_path}. Run a full optimization first."
            )
        history = np.load(history_path, allow_pickle=False)
        iters = np.asarray(history["iters"])
        energy_mean = np.asarray(history["energy_mean"])
        energy_sigma = np.asarray(history["energy_sigma"])
        energy_variance = np.asarray(history["energy_variance"])
        energy_std_local = np.asarray(history["energy_std_local"])
        energy_tau = np.asarray(history["energy_tau"])
        energy_rhat = np.asarray(history["energy_rhat"])
        update_norm_iters = np.asarray(history["update_norm_iters"])
        update_norm_values = np.asarray(history["update_norm_values"])
    else:
        if not resume_variables_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_variables_path}"
            )
        vstate.variables = flax.serialization.from_bytes(
            vstate.variables, resume_variables_path.read_bytes()
        )
        print("Loaded resume checkpoint into variational state.")

        optimizer = nk.optimizer.Sgd(
            learning_rate=resume_learning_rate
        )
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
        log.serialize(job_dir / "runtime_log")

        energy_history = log.data["Energy"]
        iters = np.asarray(energy_history.iters)
        energy_mean = np.asarray(energy_history["Mean"])
        energy_sigma = np.asarray(energy_history["Sigma"])
        energy_variance = np.asarray(energy_history["Variance"])
        energy_std_local = np.sqrt(np.maximum(np.real(energy_variance), 0.0))

        try:
            energy_tau = np.asarray(energy_history["TauCorr"])
        except Exception:
            energy_tau = np.asarray([])

        try:
            energy_rhat = np.asarray(energy_history["R_hat"])
        except Exception:
            energy_rhat = np.asarray([])

        update_norm_history = log.data.get("UpdateNormL2", None)
        if update_norm_history is not None:
            update_norm_iters = np.asarray(update_norm_history.iters)
            update_norm_values = np.asarray(update_norm_history)
        else:
            update_norm_iters = np.asarray([], dtype=np.int64)
            update_norm_values = np.asarray([], dtype=np.float64)

        np.savez(
            raw_data_dir / "optimization_history.npz",
            iters=iters,
            energy_mean=energy_mean,
            energy_sigma=energy_sigma,
            energy_variance=energy_variance,
            energy_std_local=energy_std_local,
            energy_tau=energy_tau,
            energy_rhat=energy_rhat,
            update_norm_iters=update_norm_iters,
            update_norm_values=update_norm_values,
        )

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
    ax.axhline(
        e_ref_projected,
        color="tab:green",
        ls="-.",
        lw=1.5,
        label=e_ref_projected_plot_label,
    )
    if e_ref is not None:
        ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)
        ax.set_title("Haldane model: VMC_SR vs exact and projected ED references")
    else:
        ax.set_title("Haldane model: VMC_SR vs projected ED reference")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_step.png", dpi=180)
    plt.close(fig)

    n_points = iters.size
    start_idx = max(int(math.floor(0.9 * n_points)), 0)
    if start_idx >= n_points and n_points > 0:
        start_idx = n_points - 1
    iters_final = iters[start_idx:]
    energy_mean_final = energy_mean[start_idx:]
    energy_sigma_final = energy_sigma[start_idx:]

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
        ax.axhline(
            e_ref_projected,
            color="tab:green",
            ls="-.",
            lw=1.5,
            label=e_ref_projected_plot_label,
        )
        if e_ref is not None:
            ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)
            ax.set_title("Haldane model: final 10% VMC_SR vs exact and projected ED")
        else:
            ax.set_title("Haldane model: final 10% VMC_SR vs projected ED")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Haldane model: final 10% VMC_SR energy")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_step_final10pct.png", dpi=180)
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
    fig.savefig(plots_dir / "local_energy_std_log10.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if update_norm_values.size > 0:
        ax.plot(update_norm_iters, update_norm_values, color="tab:red", lw=1.5)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Update norm not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"||$\Delta\theta$||$_2$")
    ax.set_title("Update norm")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "update_norm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if energy_tau.size > 0:
        ax.plot(iters, energy_tau, color="tab:orange", lw=1.5, label="TauCorr")
        if energy_rhat.size > 0:
            ax.plot(iters, energy_rhat, color="tab:brown", lw=1.5, label="R_hat")
        ax.legend()
        ax.set_title("Sampling diagnostics")
    else:
        ax.text(0.5, 0.5, "No extra diagnostics available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Sampling diagnostics")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "sampling_diagnostics.png", dpi=180)
    plt.close(fig)

    (plots_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
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
            "n_iter": n_iter,
            "n_samples": n_samples,
            "mc_chunk_size": mc_chunk_size,
            "n_discard_per_chain": n_discard_per_chain,
            "n_chains": n_chains,
            "sweep_size": sweep_size,
            "optimizer": f"Sgd(learning_rate={resume_learning_rate})",
            "resume_from_vstate": str(resume_variables_path),
            "diag_shift": 0.01,
            "mode": "complex",
        },
        "n_wavefunction_params": int(n_wavefunction_params),
        "final_energy_real": float(np.real(energy_mean[-1])) if energy_mean.size > 0 else None,
        "final_energy_sigma": float(energy_sigma[-1]) if energy_sigma.size > 0 else None,
        "final_update_norm": (
            float(update_norm_values[-1]) if update_norm_values.size > 0 else None
        ),
        "reference_method": e_ref_method,
        "reference_energy": None if e_ref is None else float(e_ref),
        "projected_reference_method": e_ref_projected_method,
        "projected_reference_energy": float(e_ref_projected),
        "projected_reference_sector": [int(v) for v in e_ref_projected_sector],
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if e_ref is not None:
        print(f"Exact ground-state reference [{e_ref_method}] = {e_ref:.10f}")
        print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
        print(f"Final error (VMC - reference) = {energy_mean.real[-1] - e_ref:.10f}")
    else:
        print(f"Exact ground-state reference [{e_ref_method}]")
        print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
    print(
        "Projected lowest-band reference "
        f"[{e_ref_projected_method}] = {e_ref_projected:.10f}"
    )
    print(
        "Final error (VMC - projected reference) = "
        f"{energy_mean.real[-1] - e_ref_projected:.10f}"
    )
    if update_norm_values.size > 0:
        print(f"Final update norm ||dtheta||_2 = {float(update_norm_values[-1]):.6e}")
    print(f"Final local-energy std dev = {float(energy_std_local[-1]):.10f}")


if __name__ == "__main__":
    main()
