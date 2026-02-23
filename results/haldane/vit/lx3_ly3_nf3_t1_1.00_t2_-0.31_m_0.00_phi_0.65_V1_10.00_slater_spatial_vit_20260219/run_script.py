import os
import json
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
from aidan_custom.job_script_utils import (
    build_resume_setup_signature,
    concatenate_histories,
    find_resume_source_for_setup,
    load_history_for_append,
    save_history_npz,
)
from aidan_custom.models import (
    LogSlaterSpatialViT,
    make_translation_equivariant_pair_data_from_graph,
)
from aidan_custom.optimization import (
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
)


# Written with Codex 02-19-26.
def main():
    matplotlib.use("Agg")

    job_dir = Path(__file__).resolve().parent
    raw_data_dir = job_dir / "raw_data"
    plots_dir = job_dir / "plots_optimization"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    backend = jax.default_backend()
    print("devices:", devices)
    print("default_backend:", backend)

    # Same system/Hamiltonian settings as notebooks/haldane_model.ipynb, except V1.
    Lx = 3
    Ly = 3
    n_fermions = 3
    V1 = 10.0

    t1 = 1.0
    t2 = -1.0 / (4.0 * np.cos(0.65))
    phi = 0.65
    m = 0.0

    # Same network/optimization settings as notebook, except n_iter.
    model_type = "slater_spatial_vit"
    sample_type = "FullSum"

    n_iter = 2_000
    n_samples = 1024 * 4
    n_discard_per_chain = 64
    sweep_size = 64
    n_chains = 16

    vit_num_layers = 2
    vit_d_model = 16
    vit_n_heads = 2
    vit_mlp_hidden_factor = 2
    vit_output_hidden_dim = 16
    vit_xi_epsilon = 1.0e-6
    slater_init_mode = "noninteracting"
    optimizer_name = "Sgd(linear_schedule(0.05->0.05))"
    diag_shift = 0.01
    optimization_mode = "complex"

    resume_setup_signature = build_resume_setup_signature(
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
        V1=V1,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        model_type=model_type,
        sample_type=sample_type,
        vit_num_layers=vit_num_layers,
        vit_d_model=vit_d_model,
        vit_n_heads=vit_n_heads,
        vit_mlp_hidden_factor=vit_mlp_hidden_factor,
        vit_output_hidden_dim=vit_output_hidden_dim,
        vit_xi_epsilon=vit_xi_epsilon,
        slater_init_mode=slater_init_mode,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        sweep_size=sweep_size,
        n_chains=n_chains,
        optimizer_name=optimizer_name,
        diag_shift=diag_shift,
        mode=optimization_mode,
    )

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
    ham_sr = ham.to_fermionoperator2nd()

    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        n_chains=n_chains,
        sweep_size=sweep_size,
    )

    if model_type != "slater_spatial_vit":
        raise ValueError(f"Unsupported model_type={model_type!r} in this script.")

    pair_classes, pair_distances, _ = make_translation_equivariant_pair_data_from_graph(graph)
    pair_classes_hashable = tuple(tuple(int(v) for v in row) for row in pair_classes)
    pair_distances_hashable = tuple(float(v) for v in pair_distances)

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

    model = LogSlaterSpatialViT(
        hilbert=hi,
        num_layers=vit_num_layers,
        d_model=vit_d_model,
        n_heads=vit_n_heads,
        pair_classes=pair_classes_hashable,
        pair_distances=pair_distances_hashable,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=slater_initial_m_orbitals,
        mlp_hidden_factor=vit_mlp_hidden_factor,
        output_hidden_dim=vit_output_hidden_dim,
        xi_epsilon=vit_xi_epsilon,
    )

    if sample_type == "MC":
        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
        )
    elif sample_type == "FullSum":
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        raise ValueError(f"Unknown sample_type={sample_type!r}")

    resume_job_dir, resume_summary = find_resume_source_for_setup(
        job_dir=job_dir,
        results_dir=job_dir.parent,
        resume_setup_signature=resume_setup_signature,
    )
    previous_history: dict[str, np.ndarray] | None = None

    if resume_job_dir is not None:
        resume_ckpt = resume_job_dir / "vstate_variables.mpack"
        vstate.variables = nk.experimental.vqs.variables_from_file(
            str(resume_ckpt), vstate.variables
        )
        previous_history = load_history_for_append(resume_job_dir)
        print(f"Resuming from existing optimized state: {resume_ckpt}")
    else:
        print("No prior matching setup found; starting from ordinary initialization.")

    n_wavefunction_params = nk.jax.tree_size(vstate.parameters)
    print(f"total # of wavefunction parameters: {n_wavefunction_params}")

    optimizer = nk.optimizer.Sgd(
        learning_rate=optax.linear_schedule(0.05, 0.05, n_iter)
    )
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        mode=optimization_mode,
    )

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=log_optimization_diagnostics)

    (job_dir / "vstate_variables.mpack").write_bytes(
        flax.serialization.to_bytes(vstate.variables)
    )
    param_leaves, param_treedef = jax.tree_util.tree_flatten(vstate.parameters)
    np.savez(job_dir / "vstate_parameters_leaves.npz", *[np.asarray(x) for x in param_leaves])
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

    current_history = {
        "iters": np.asarray(iters),
        "energy_mean": np.asarray(energy_mean),
        "energy_sigma": np.asarray(energy_sigma),
        "energy_variance": np.asarray(energy_variance),
        "energy_std_local": np.asarray(energy_std_local),
        "energy_tau": np.asarray(energy_tau),
        "energy_rhat": np.asarray(energy_rhat),
        "update_norm_iters": np.asarray(update_norm_iters),
        "update_norm_values": np.asarray(update_norm_values),
    }
    contiguous_history = concatenate_histories(previous_history, current_history)

    save_history_npz(raw_data_dir / "optimization_history.npz", current_history)
    save_history_npz(raw_data_dir / "optimization_history_contiguous.npz", contiguous_history)

    iters_plot = contiguous_history["iters"]
    energy_mean_plot = contiguous_history["energy_mean"]
    energy_sigma_plot = contiguous_history["energy_sigma"]
    energy_std_local_plot = contiguous_history["energy_std_local"]
    energy_tau_plot = contiguous_history["energy_tau"]
    energy_rhat_plot = contiguous_history["energy_rhat"]
    update_norm_iters_plot = contiguous_history["update_norm_iters"]
    update_norm_values_plot = contiguous_history["update_norm_values"]

    e_ref, e_ref_plot_label, e_ref_method = exact_reference_ground_state_energy(
        hamiltonian=ham,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters_plot, np.real(energy_mean_plot), lw=1.8, color="tab:blue", label="VMC_SR")
    ax.fill_between(
        iters_plot,
        np.real(energy_mean_plot) - energy_sigma_plot,
        np.real(energy_mean_plot) + energy_sigma_plot,
        color="tab:blue",
        alpha=0.2,
        linewidth=0,
    )
    ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)
    ax.set_title("Haldane model: VMC_SR vs exact ground-state energy")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_step.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        iters_plot,
        np.log10(np.maximum(energy_std_local_plot, 1e-30)),
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
    if update_norm_values_plot.size > 0:
        ax.plot(update_norm_iters_plot, update_norm_values_plot, color="tab:red", lw=1.5)
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
    if energy_tau_plot.size > 0:
        ax.plot(iters_plot, energy_tau_plot, color="tab:orange", lw=1.5, label="TauCorr")
        if energy_rhat_plot.size > 0:
            ax.plot(iters_plot, energy_rhat_plot, color="tab:brown", lw=1.5, label="R_hat")
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

    summary = {
        "job_dir": str(job_dir),
        "device": str(devices),
        "backend": backend,
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
            "model_type": model_type,
            "sample_type": sample_type,
            "vit_num_layers": vit_num_layers,
            "vit_d_model": vit_d_model,
            "vit_n_heads": vit_n_heads,
            "vit_mlp_hidden_factor": vit_mlp_hidden_factor,
            "vit_output_hidden_dim": vit_output_hidden_dim,
            "vit_xi_epsilon": vit_xi_epsilon,
            "slater_init_mode": slater_init_mode,
            "n_wavefunction_params": int(n_wavefunction_params),
        },
        "optimization": {
            "n_iter": n_iter,
            "n_samples": n_samples,
            "n_discard_per_chain": n_discard_per_chain,
            "sweep_size": sweep_size,
            "n_chains": n_chains,
            "optimizer": optimizer_name,
            "diag_shift": diag_shift,
            "mode": optimization_mode,
        },
        "resume_setup": resume_setup_signature,
        "resume": {
            "used": resume_job_dir is not None,
            "source_job_dir": None if resume_job_dir is None else str(resume_job_dir),
            "source_summary_file": (
                None if resume_job_dir is None else str(resume_job_dir / "summary.json")
            ),
            "source_n_iter": (
                None
                if resume_summary is None
                else (
                    resume_summary.get("optimization", {}).get("n_iter")
                    if isinstance(resume_summary.get("optimization"), dict)
                    else None
                )
            ),
            "previous_total_steps": (
                0
                if previous_history is None or previous_history["iters"].size == 0
                else int(previous_history["iters"][-1]) + 1
            ),
            "contiguous_total_steps": (
                0
                if contiguous_history["iters"].size == 0
                else int(contiguous_history["iters"][-1]) + 1
            ),
        },
        "results": {
            "final_energy_real": float(np.real(energy_mean[-1])),
            "final_energy_sigma": float(energy_sigma[-1]),
            "final_local_energy_std": float(energy_std_local[-1]),
            "final_update_norm": float(update_norm_values[-1]) if update_norm_values.size > 0 else None,
            "reference_method": e_ref_method,
            "reference_energy": float(e_ref),
            "error_vs_reference": float(np.real(energy_mean[-1]) - e_ref),
        },
        "artifacts": {
            "vstate_variables": "vstate_variables.mpack",
            "vstate_parameters_leaves": "vstate_parameters_leaves.npz",
            "vstate_parameters_treedef": "vstate_parameters_treedef.txt",
            "runtime_log": "runtime_log.json",
            "raw_data_latest_round": "raw_data/optimization_history.npz",
            "raw_data_contiguous": "raw_data/optimization_history_contiguous.npz",
            "plots_dir": "plots_optimization",
            "script": "run_script.py",
        },
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("RESULTS_START")
    print(f"job_dir={job_dir}")
    print(f"final_energy_real={float(np.real(energy_mean[-1])):.12f}")
    print(f"reference_energy={float(e_ref):.12f}")
    print(f"error_vs_reference={float(np.real(energy_mean[-1]) - e_ref):.12f}")
    print(
        "resume_source="
        + ("None" if resume_job_dir is None else str(resume_job_dir))
    )
    print(
        "contiguous_total_steps="
        + str(
            0
            if contiguous_history["iters"].size == 0
            else int(contiguous_history["iters"][-1]) + 1
        )
    )
    print(f"summary_file={job_dir / 'summary.json'}")
    print("RESULTS_END")


if __name__ == "__main__":
    main()
