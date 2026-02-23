import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import (
    LogSlaterBoseFormer,
    make_supercell_reciprocal_vectors_from_graph,
)


def collect_timer_records(timer, root_total, parent_total, path_prefix=""):
    # Written with Codex 02-20-26.
    records = []
    for name, sub_timer in sorted(
        timer.sub_timers.items(), key=lambda item: item[1].total, reverse=True
    ):
        path = f"{path_prefix}/{name}" if path_prefix else name
        seconds = float(sub_timer.total)
        record = {
            "path": path,
            "name": name,
            "seconds": seconds,
            "percent_of_total": 100.0 * seconds / root_total if root_total > 0.0 else 0.0,
            "percent_of_parent": (
                100.0 * seconds / parent_total if parent_total > 0.0 else 0.0
            ),
        }
        records.append(record)
        records.extend(
            collect_timer_records(
                sub_timer,
                root_total=root_total,
                parent_total=seconds,
                path_prefix=path,
            )
        )
    return records


def main():
    # Written with Codex 02-20-26.
    devices = jax.devices()
    backend = jax.default_backend()
    print("devices:", devices)
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend}. Refusing to run benchmark on CPU."
        )

    benchmark_dir = Path(__file__).resolve().parent
    job_dir = benchmark_dir.parent
    raw_data_dir = benchmark_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Match the production job configuration.
    t1 = 1.0
    phi = 0.65
    t2 = -1.0 / (4.0 * np.cos(phi))
    m = 0.0
    Lx = 4
    Ly = 3
    V1 = 10.0
    n_fermions = 4
    n_iter_full_job = 1_000

    # Use a short run for timing.
    warmup_n_iter = 2
    timed_n_iter = 20

    # Model parameters from the production job.
    boseformer_num_layers = 2
    boseformer_d_model = 32
    boseformer_n_heads = 2
    boseformer_mlp_hidden_factor = 2

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
    print(f"n_sites={graph.n_nodes}, n_fermions={hi.n_fermions}")
    print(f"hilbert={hi}")
    print(f"max_conn_size (original operator)={ham.max_conn_size}")

    if isinstance(ham, nk.operator.FermionOperator2nd):
        ham_fermion = ham
    else:
        ham_fermion = ham.to_fermionoperator2nd()

    requested_operator = "ParticleNumberAndSpinConservingFermioperator2nd"
    conversion_note = None
    try:
        ham_sr = (
            nk.experimental.operator.ParticleNumberAndSpinConservingFermioperator2nd
            .from_fermionoperator2nd(ham_fermion)
        )
    except AssertionError:
        # Spin-sector-conserving operator requires n_spin_subsectors >= 2.
        ham_sr = (
            nk.experimental.operator.ParticleNumberConservingFermioperator2nd
            .from_fermionoperator2nd(ham_fermion)
        )
        conversion_note = (
            "Fallback to ParticleNumberConservingFermioperator2nd because "
            "this Hilbert space has no explicit spin subsectors "
            "(n_spin_subsectors < 2)."
        )
    print(f"using operator type: {type(ham_sr).__name__}")
    if conversion_note is not None:
        print(conversion_note)
    print(f"max_conn_size (conserving operator)={ham_sr.max_conn_size}")
    hash_fix_applied = False
    try:
        _ = hash(ham_sr)
    except TypeError:
        # VMC_SR uses to_sparse_cached, which requires hashable operators.
        # This class can fail to hash due to internal mutable dict fields.
        type(ham_sr).__hash__ = object.__hash__
        _ = hash(ham_sr)
        hash_fix_applied = True
        print("Applied runtime hash workaround for to_sparse_cached compatibility.")

    g_vectors = make_supercell_reciprocal_vectors_from_graph(graph)
    positions_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(graph.positions, dtype=np.float64)
    )
    g_vectors_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(g_vectors, dtype=np.float64)
    )

    model = LogSlaterBoseFormer(
        hilbert=hi,
        positions=positions_hashable,
        g_vectors=g_vectors_hashable,
        num_layers=boseformer_num_layers,
        d_model=boseformer_d_model,
        n_heads=boseformer_n_heads,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=None,
        boseformer_param_dtype=jnp.float64,
        mlp_hidden_factor=boseformer_mlp_hidden_factor,
    )

    vstate = nk.vqs.FullSumState(hi, model)
    n_wavefunction_params = nk.jax.tree_size(vstate.parameters)
    print(f"total # of wavefunction parameters: {n_wavefunction_params}")

    optimizer = nk.optimizer.Sgd(
        learning_rate=optax.linear_schedule(0.05, 0.005, n_iter_full_job)
    )
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=0.01,
        mode="complex",
    )

    print(f"Warmup run (n_iter={warmup_n_iter})...")
    warmup_log = nk.logging.RuntimeLog()
    driver.run(n_iter=warmup_n_iter, out=warmup_log, show_progress=False)

    print(f"Timed run (n_iter={timed_n_iter}, timeit=True)...")
    timed_log = nk.logging.RuntimeLog()
    driver.run(
        n_iter=timed_n_iter,
        out=timed_log,
        show_progress=False,
        timeit=True,
    )

    timer = driver._timer
    total_seconds = float(timer.total)
    top_level = [
        {"name": name, "seconds": float(sub.total)}
        for name, sub in sorted(
            timer.sub_timers.items(), key=lambda item: item[1].total, reverse=True
        )
    ]
    for entry in top_level:
        entry["percent_of_total"] = (
            100.0 * entry["seconds"] / total_seconds if total_seconds > 0.0 else 0.0
        )

    flat_records = collect_timer_records(
        timer, root_total=total_seconds, parent_total=total_seconds
    )
    flat_records_sorted = sorted(
        flat_records, key=lambda record: record["percent_of_total"], reverse=True
    )
    dominant_top_level = top_level[0] if len(top_level) > 0 else None
    dominant_scope = flat_records_sorted[0] if len(flat_records_sorted) > 0 else None

    summary = {
        "job_dir": str(job_dir),
        "benchmark_dir": str(benchmark_dir),
        "backend": backend,
        "requested_operator_type": requested_operator,
        "operator_type": type(ham_sr).__name__,
        "conversion_note": conversion_note,
        "hash_workaround_applied": hash_fix_applied,
        "max_conn_size_original": int(ham.max_conn_size),
        "max_conn_size_conserving": int(ham_sr.max_conn_size),
        "warmup_n_iter": warmup_n_iter,
        "timed_n_iter": timed_n_iter,
        "n_wavefunction_params": int(n_wavefunction_params),
        "timer_total_seconds": total_seconds,
        "top_level_scopes": top_level,
        "all_scopes": flat_records_sorted,
        "dominant_top_level_scope": dominant_top_level,
        "dominant_scope": dominant_scope,
    }

    (raw_data_dir / "timing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    timed_log.serialize(raw_data_dir / "runtime_log_benchmark")
    (benchmark_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"Total timed window: {total_seconds:.6f} s")
    print("Top-level timing scopes:")
    for entry in top_level:
        print(
            f"  {entry['name']}: {entry['seconds']:.6f} s "
            f"({entry['percent_of_total']:.2f}%)"
        )
    if dominant_scope is not None:
        print(
            "Dominant scope overall: "
            f"{dominant_scope['path']} "
            f"({dominant_scope['seconds']:.6f} s, "
            f"{dominant_scope['percent_of_total']:.2f}%)"
        )
    print(f"Saved benchmark outputs under: {benchmark_dir}")


if __name__ == "__main__":
    main()
