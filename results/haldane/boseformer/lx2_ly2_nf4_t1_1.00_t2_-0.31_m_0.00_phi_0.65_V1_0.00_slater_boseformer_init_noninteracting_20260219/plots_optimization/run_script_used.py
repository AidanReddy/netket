import json
from pathlib import Path

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
from aidan_custom.optimization import exact_reference_ground_state_energy


def _hashable_matrix(matrix: np.ndarray | jax.Array) -> tuple[tuple[float, ...], ...]:
    # Written with Codex 02-19-26.
    arr = np.asarray(matrix)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def main():
    # Written with Codex 02-19-26.
    matplotlib.use("Agg")

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

    slater_init_mode = "noninteracting"  # options: "noninteracting", "random"

    num_layers = 2
    d_model = 32
    n_heads = 4
    mlp_hidden_factor = 4

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

    vstate = nk.vqs.FullSumState(hi, model)
    initial_energy_stats = vstate.expect(ham_sr)
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

    iters = np.asarray([0], dtype=np.int64)
    energy_mean = np.asarray([initial_energy], dtype=np.float64)
    energy_sigma = np.asarray([0.0], dtype=np.float64)
    final_energy = initial_energy

    np.savez(
        raw_data_dir / "energy_history.npz",
        iters=iters,
        energy_mean=energy_mean,
        energy_sigma=energy_sigma,
    )

    (job_dir / "vstate_variables.mpack").write_bytes(
        flax.serialization.to_bytes(vstate.variables)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, np.real(energy_mean), color="tab:blue", lw=1.7, label="Initialization")
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
        "optimization": {"mode": "none (initialization-only non-interacting recovery check)"},
        "energies": {
            "reference_energy": float(e_ref),
            "reference_method": e_ref_method,
            "initial_energy": initial_energy,
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
    print("final_energy:", final_energy, "abs_error:", final_abs_error)
    print("recovered_within_tolerance:", recovered)


if __name__ == "__main__":
    main()
