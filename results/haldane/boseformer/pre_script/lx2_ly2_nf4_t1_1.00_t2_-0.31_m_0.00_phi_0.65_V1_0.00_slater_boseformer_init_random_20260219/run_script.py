import os
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from aidan_custom.optimization import (
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
)


def _hashable_matrix(matrix: np.ndarray | jax.Array) -> tuple[tuple[float, ...], ...]:
    # Written with Codex 02-19-26.
    arr = np.asarray(matrix)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def main():
    # Written with Codex 02-19-26.
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

    Lx = 2
    Ly = 2
    n_fermions = 4
    V1 = 0.0

    t1 = 1.0
    phi = 0.65
    t2 = -1.0 / (4.0 * np.cos(phi))
    m = 0.0

    slater_init_mode = "random"  # options: "noninteracting", "random"

    num_layers = 2
    d_model = 32
    n_heads = 4
    mlp_hidden_factor = 4

    n_iter = 80
    learning_rate = 0.02
    diag_shift = 0.05
    optimization_mode = "complex"

    recovery_tolerance = 1.0e-8

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
    elif slater_init_mode != "random":
        raise ValueError(f"Unknown slater_init_mode={slater_init_mode!r}")

    g_vectors = make_supercell_reciprocal_vectors_from_graph(graph)
    model = LogSlaterBoseFormer(
        hilbert=hi,
        positions=_hashable_matrix(np.asarray(graph.positions, dtype=np.float64)),
        g_vectors=_hashable_matrix(np.asarray(g_vectors, dtype=np.float64)),
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        mlp_hidden_factor=mlp_hidden_factor,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=slater_initial_m_orbitals,
        boseformer_param_dtype=jnp.float64,
    )

    vstate_train = nk.vqs.FullSumState(hi, model)
    initial_energy_stats = vstate_train.expect(ham_sr)
    initial_energy = float(np.real(np.asarray(initial_energy_stats.mean)))

    e_ref, e_ref_label, e_ref_method = exact_reference_ground_state_energy(
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

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate_train,
        diag_shift=diag_shift,
        mode=optimization_mode,
    )
    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log, callback=log_optimization_diagnostics)

    energy_history = log.data["Energy"]
    iters = np.asarray(energy_history.iters)
    energy_mean = np.asarray(energy_history["Mean"])
    energy_sigma = np.asarray(energy_history["Sigma"])
    driver_final_energy_estimate = float(np.real(energy_mean[-1]))

    final_energy_stats = vstate_train.expect(ham_sr)
    final_energy = float(np.real(np.asarray(final_energy_stats.mean)))

    np.savez(
        raw_data_dir / "energy_history.npz",
        iters=iters,
        energy_mean=energy_mean,
        energy_sigma=energy_sigma,
        driver_final_energy_estimate=driver_final_energy_estimate,
    )
    log.serialize(job_dir / "runtime_log")

    (job_dir / "vstate_variables.mpack").write_bytes(
        flax.serialization.to_bytes(vstate_train.variables)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.real(energy_mean), color="tab:blue", lw=1.7, label="VMC_SR (FullSum)")
    ax.fill_between(
        iters,
        np.real(energy_mean) - energy_sigma,
        np.real(energy_mean) + energy_sigma,
        color="tab:blue",
        alpha=0.22,
        linewidth=0,
    )
    ax.axhline(e_ref, color="black", ls="--", lw=1.4, label=e_ref_label)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy")
    ax.set_title("LogSlaterBoseFormer: non-interacting recovery test")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_step.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, energy_sigma, color="tab:purple", lw=1.6)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy sigma")
    ax.set_title("LogSlaterBoseFormer: energy sigma history")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_sigma_vs_step.png", dpi=180)
    plt.close(fig)

    (plots_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    initial_abs_error = float(abs(initial_energy - e_ref))
    final_abs_error = float(abs(final_energy - e_ref))
    recovered = final_abs_error <= recovery_tolerance

    summary = {
        "job_dir": str(job_dir),
        "model": "LogSlaterBoseFormer",
        "system": {
            "Lx": Lx,
            "Ly": Ly,
            "n_fermions": n_fermions,
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "m": m,
            "V1": V1,
        },
        "network": {
            "num_layers": num_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "mlp_hidden_factor": mlp_hidden_factor,
            "slater_init_mode": slater_init_mode,
        },
        "optimization": {
            "mode": optimization_mode,
            "driver": "VMC_SR",
            "sample_type": "FullSum",
            "n_iter": n_iter,
            "learning_rate": learning_rate,
            "diag_shift": diag_shift,
        },
        "energies": {
            "reference_energy": float(e_ref),
            "reference_method": e_ref_method,
            "initial_energy": initial_energy,
            "driver_final_energy_estimate": driver_final_energy_estimate,
            "final_energy": final_energy,
            "initial_abs_error": initial_abs_error,
            "final_abs_error": final_abs_error,
            "recovery_tolerance": recovery_tolerance,
            "recovered_within_tolerance": bool(recovered),
        },
    }

    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("backend:", jax.default_backend())
    print("reference_energy:", e_ref)
    print("initial_energy:", initial_energy, "abs_error:", initial_abs_error)
    print("driver_final_energy_estimate:", driver_final_energy_estimate)
    print("final_energy:", final_energy, "abs_error:", final_abs_error)
    print("recovered_within_tolerance:", recovered)


if __name__ == "__main__":
    main()
