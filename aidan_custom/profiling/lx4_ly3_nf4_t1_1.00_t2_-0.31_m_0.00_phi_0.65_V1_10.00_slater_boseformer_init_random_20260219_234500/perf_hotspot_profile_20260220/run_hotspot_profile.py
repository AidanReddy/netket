import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
from netket import jax as nkjax
import numpy as np
import optax

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import (
    LogSlaterBoseFormer,
    make_supercell_reciprocal_vectors_from_graph,
)
from netket._src.ngd.sr import _compute_sr_update
from netket._src.ngd.sr_srt_common import _prepare_input, _prepare_weights
from netket._src.ngd.srt import _compute_srt_update


@jax.jit
def build_sr_force_vector(O_L, dv):
    # Written with Codex 02-20-26.
    return O_L.T @ dv


@jax.jit
def build_sr_qgt_matrix(O_L):
    # Written with Codex 02-20-26.
    return O_L.T @ O_L


@jax.jit
def build_srt_ntk_matrix(O_L):
    # Written with Codex 02-20-26.
    return O_L @ O_L.T


def sync_output(output):
    # Written with Codex 02-20-26.
    try:
        return jax.block_until_ready(output)
    except Exception:
        return output


def summarize_array(name, value):
    # Written with Codex 02-20-26.
    shape = list(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    value_type = type(value).__name__
    sharding = None
    device = None
    if hasattr(value, "sharding"):
        sharding = str(value.sharding)
    if hasattr(value, "device"):
        try:
            device = str(value.device)
        except Exception:
            device = None
    return {
        "name": name,
        "type": value_type,
        "dtype": dtype,
        "shape": shape,
        "device": device,
        "sharding": sharding,
    }


def time_block(name, fn, warmup, repeat, keep_output=True):
    # Written with Codex 02-20-26.
    last_output = None
    for _ in range(warmup):
        out = fn()
        sync_output(out)
        if keep_output:
            last_output = out
        else:
            last_output = None
            del out

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        sync_output(out)
        times.append(time.perf_counter() - t0)
        if keep_output:
            last_output = out
        else:
            last_output = None
            del out

    arr = np.asarray(times, dtype=np.float64)
    summary = {
        "name": name,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "times_s": [float(x) for x in arr],
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=0)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }
    return summary, last_output


def plot_bar(names, values, title, xlabel, output_path):
    # Written with Codex 02-20-26.
    values = np.asarray(values, dtype=np.float64)
    fig_height = max(3.2, 0.52 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(names))
    ax.barh(y, values, color="tab:blue", alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x")
    ref = float(values.max()) if values.size > 0 else 0.0
    for i, value in enumerate(values):
        x_pos = value + (0.01 * ref if ref > 0.0 else 0.02)
        ax.text(x_pos, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    # Written with Codex 02-20-26.
    matplotlib.use("Agg")

    backend = jax.default_backend()
    print("devices:", jax.devices())
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
    diag_shift = 0.01
    mode = "complex"

    # Benchmark controls.
    warmup = 1
    repeat_heavy = 2
    repeat_medium = 3
    repeat_light = 5

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
    print(f"max_conn_size={ham.max_conn_size}")

    g_vectors = make_supercell_reciprocal_vectors_from_graph(graph)
    positions_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(graph.positions, dtype=np.float64)
    )
    g_vectors_hashable = tuple(
        tuple(float(v) for v in row)
        for row in np.asarray(g_vectors, dtype=np.float64)
    )

    if isinstance(ham, nk.operator.FermionOperator2nd):
        ham_sr = ham
    else:
        ham_sr = ham.to_fermionoperator2nd()

    model = LogSlaterBoseFormer(
        hilbert=hi,
        positions=positions_hashable,
        g_vectors=g_vectors_hashable,
        num_layers=2,
        d_model=32,
        n_heads=2,
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=None,
        boseformer_param_dtype=jnp.float64,
        mlp_hidden_factor=2,
    )
    vstate = nk.vqs.FullSumState(hi, model)
    n_wavefunction_params = int(nk.jax.tree_size(vstate.parameters))
    print(f"total # of wavefunction parameters: {n_wavefunction_params}")

    optimizer = nk.optimizer.Sgd(
        learning_rate=optax.linear_schedule(0.05, 0.005, n_iter_full_job)
    )
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        mode=mode,
    )
    print(f"use_ntk={driver.use_ntk}, on_the_fly={driver.on_the_fly}, mode={driver.mode}")

    O_sparse = nk.operator.to_sparse_cached(ham_sr)
    samples = vstate._all_states
    params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters
    )
    _, unravel_params_fn = ravel_pytree(vstate.parameters)

    # Reference tensors used in component-level profiling.
    vstate.reset()
    psi_ref = vstate.to_array()
    Opsi_ref = O_sparse @ psi_ref
    local_ref = Opsi_ref / psi_ref
    weights_ref = np.abs(psi_ref) ** 2
    pdf_ref, mass_ref = _prepare_weights(weights_ref, samples.shape[0])
    jac_ref = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        samples,
        vstate.model_state,
        mode=driver.mode,
        dense=True,
        center=True,
        chunk_size=driver.chunk_size_bwd,
        pdf=pdf_ref,
    )
    sync_output(jac_ref)
    O_L_ref, dv_ref = _prepare_input(
        jac_ref,
        local_ref,
        mode=driver.mode,
        scaling_factor=mass_ref,
    )
    sync_output((O_L_ref, dv_ref))

    if driver.use_ntk:
        compute_update_fn = _compute_srt_update
        update_block_name = "compute_srt_update_total"
        matrix_block_name = "srt_build_ntk_matrix_O_O_T"
        matrix_block_fn = lambda: build_srt_ntk_matrix(O_L_ref)
    else:
        compute_update_fn = _compute_sr_update
        update_block_name = "compute_sr_update_total"
        matrix_block_name = "sr_build_qgt_matrix_O_T_O"
        matrix_block_fn = lambda: build_sr_qgt_matrix(O_L_ref)

    updates_ref, _, _ = compute_update_fn(
        O_L_ref,
        dv_ref,
        diag_shift=diag_shift,
        solver_fn=driver._linear_solver,
        mode=driver.mode,
        proj_reg=None,
        momentum=None,
        old_updates=None,
        params_structure=params_structure,
    )
    sync_output(updates_ref)

    # Benchmark blocks.
    records = []
    outputs_info = []

    record, _ = time_block(
        "forward_backward_full",
        lambda: driver._forward_and_backward(),
        warmup=warmup,
        repeat=repeat_heavy,
    )
    records.append(record)

    record, _ = time_block(
        "to_array_after_reset",
        lambda: (vstate.reset(), vstate.to_array())[1],
        warmup=warmup,
        repeat=repeat_medium,
    )
    records.append(record)

    record, Opsi_last = time_block(
        "sparse_matvec_O_times_psi",
        lambda: O_sparse @ psi_ref,
        warmup=warmup,
        repeat=repeat_light,
    )
    records.append(record)

    record, local_last = time_block(
        "local_energy_postprocess",
        lambda: Opsi_ref / psi_ref,
        warmup=warmup,
        repeat=repeat_light,
    )
    records.append(record)

    record, pdf_mass_last = time_block(
        "prepare_weights_pdf_mass",
        lambda: _prepare_weights(weights_ref, samples.shape[0]),
        warmup=warmup,
        repeat=repeat_light,
    )
    records.append(record)

    record, jac_last = time_block(
        "jacobian_dense_centered",
        lambda: nkjax.jacobian(
            vstate._apply_fun,
            vstate.parameters,
            samples,
            vstate.model_state,
            mode=driver.mode,
            dense=True,
            center=True,
            chunk_size=driver.chunk_size_bwd,
            pdf=pdf_ref,
        ),
        warmup=warmup,
        repeat=repeat_heavy,
    )
    records.append(record)

    record, prep_last = time_block(
        "prepare_input_O_L_dv",
        lambda: _prepare_input(
            jac_ref,
            local_ref,
            mode=driver.mode,
            scaling_factor=mass_ref,
        ),
        warmup=warmup,
        repeat=repeat_medium,
    )
    records.append(record)

    record, update_last = time_block(
        update_block_name,
        lambda: compute_update_fn(
            O_L_ref,
            dv_ref,
            diag_shift=diag_shift,
            solver_fn=driver._linear_solver,
            mode=driver.mode,
            proj_reg=None,
            momentum=None,
            old_updates=None,
            params_structure=params_structure,
        ),
        warmup=warmup,
        repeat=repeat_heavy,
    )
    records.append(record)

    record, _ = time_block(
        "unravel_updates_to_pytree",
        lambda: unravel_params_fn(updates_ref),
        warmup=warmup,
        repeat=repeat_light,
    )
    records.append(record)

    record, _ = time_block(
        matrix_block_name,
        matrix_block_fn,
        warmup=warmup,
        repeat=repeat_heavy,
        keep_output=False,
    )
    records.append(record)

    if not driver.use_ntk:
        record, _ = time_block(
            "sr_build_force_vector_O_T_dv",
            lambda: build_sr_force_vector(O_L_ref, dv_ref),
            warmup=warmup,
            repeat=repeat_medium,
            keep_output=False,
        )
        records.append(record)

    outputs_info.extend(
        [
            summarize_array("psi_ref", psi_ref),
            summarize_array("Opsi_ref", Opsi_ref),
            summarize_array("local_ref", local_ref),
            summarize_array("weights_ref", weights_ref),
            summarize_array("pdf_ref", pdf_ref),
            summarize_array("samples", samples),
            summarize_array("jac_ref", jac_ref),
            summarize_array("O_L_ref", O_L_ref),
            summarize_array("dv_ref", dv_ref),
            summarize_array("updates_ref", updates_ref),
            summarize_array("jac_last", jac_last),
            summarize_array("prep_last_O_L", prep_last[0]),
            summarize_array("update_last_updates", update_last[0]),
            summarize_array("Opsi_last", Opsi_last),
            summarize_array("local_last", local_last),
            summarize_array("pdf_mass_last_pdf", pdf_mass_last[0]),
        ]
    )

    by_name = {record["name"]: record for record in records}
    fb_mean = by_name["forward_backward_full"]["mean_s"]
    update_mean = by_name[update_block_name]["mean_s"]

    for record in records:
        record["pct_of_forward_backward"] = (
            100.0 * record["mean_s"] / fb_mean if fb_mean > 0.0 else 0.0
        )
        record["pct_of_update_block"] = (
            100.0 * record["mean_s"] / update_mean if update_mean > 0.0 else 0.0
        )

    top_fb_components = [
        "to_array_after_reset",
        "sparse_matvec_O_times_psi",
        "local_energy_postprocess",
        "jacobian_dense_centered",
        "prepare_input_O_L_dv",
        update_block_name,
        "unravel_updates_to_pytree",
    ]
    fb_names = top_fb_components
    fb_values = [by_name[name]["mean_s"] for name in top_fb_components]
    fb_values_pct = [by_name[name]["pct_of_forward_backward"] for name in top_fb_components]

    if driver.use_ntk:
        matrix_mean = by_name[matrix_block_name]["mean_s"]
        remaining_mean = update_mean - matrix_mean
        update_internal = [
            {
                "name": matrix_block_name,
                "mean_s": float(matrix_mean),
                "pct_of_update_block": float(
                    100.0 * matrix_mean / update_mean if update_mean > 0.0 else 0.0
                ),
            },
            {
                "name": "srt_remaining_solve_project_pack_estimate",
                "mean_s": float(remaining_mean),
                "pct_of_update_block": float(
                    100.0 * remaining_mean / update_mean if update_mean > 0.0 else 0.0
                ),
            },
        ]
    else:
        matrix_mean = by_name[matrix_block_name]["mean_s"]
        force_mean = by_name["sr_build_force_vector_O_T_dv"]["mean_s"]
        remaining_mean = update_mean - matrix_mean - force_mean
        update_internal = [
            {
                "name": "sr_build_force_vector_O_T_dv",
                "mean_s": float(force_mean),
                "pct_of_update_block": float(
                    100.0 * force_mean / update_mean if update_mean > 0.0 else 0.0
                ),
            },
            {
                "name": matrix_block_name,
                "mean_s": float(matrix_mean),
                "pct_of_update_block": float(
                    100.0 * matrix_mean / update_mean if update_mean > 0.0 else 0.0
                ),
            },
            {
                "name": "sr_remaining_shift_solve_pack_estimate",
                "mean_s": float(remaining_mean),
                "pct_of_update_block": float(
                    100.0 * remaining_mean / update_mean if update_mean > 0.0 else 0.0
                ),
            },
        ]

    update_names = [item["name"] for item in update_internal]
    update_values = [item["mean_s"] for item in update_internal]
    update_values_pct = [item["pct_of_update_block"] for item in update_internal]

    summary = {
        "job_dir": str(job_dir),
        "benchmark_dir": str(benchmark_dir),
        "backend": backend,
        "driver_use_ntk": bool(driver.use_ntk),
        "driver_on_the_fly": bool(driver.on_the_fly),
        "driver_mode": str(driver.mode),
        "diag_shift": float(diag_shift),
        "n_wavefunction_params": n_wavefunction_params,
        "n_samples_fullsum": int(samples.shape[0]),
        "records": records,
        "forward_backward_mean_s": float(fb_mean),
        "update_block_name": update_block_name,
        "update_block_mean_s": float(update_mean),
        "top_forward_backward_components": [
            {
                "name": name,
                "mean_s": float(value),
                "pct_of_forward_backward": float(pct),
            }
            for name, value, pct in zip(fb_names, fb_values, fb_values_pct)
        ],
        "update_internal_components": update_internal,
        "array_metadata": outputs_info,
    }

    (raw_data_dir / "hotspot_timing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_bar(
        fb_names,
        fb_values,
        "Fine-grained timing: forward/backward components",
        "Mean time per call (s)",
        benchmark_dir / "hotspot_forward_backward_components_seconds.png",
    )
    plot_bar(
        fb_names,
        fb_values_pct,
        "Fine-grained timing: % of full forward/backward",
        "Percent of forward_backward_full mean (%)",
        benchmark_dir / "hotspot_forward_backward_components_percent.png",
    )
    plot_bar(
        update_names,
        update_values,
        f"Fine-grained timing: {update_block_name} internals",
        "Mean time per call (s)",
        benchmark_dir / "hotspot_update_internal_seconds.png",
    )
    plot_bar(
        update_names,
        update_values_pct,
        f"Fine-grained timing: % of {update_block_name}",
        "Percent of update block mean (%)",
        benchmark_dir / "hotspot_update_internal_percent.png",
    )

    (benchmark_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"forward_backward_full mean: {fb_mean:.6f} s")
    print(f"{update_block_name} mean: {update_mean:.6f} s")
    print("Top forward/backward component shares:")
    for name, value, pct in zip(fb_names, fb_values, fb_values_pct):
        print(f"  {name}: {value:.6f} s ({pct:.2f}%)")
    print(f"{update_block_name} internal shares:")
    for name, value, pct in zip(update_names, update_values, update_values_pct):
        print(f"  {name}: {value:.6f} s ({pct:.2f}%)")
    print(f"Saved hotspot profiling outputs under: {benchmark_dir}")


if __name__ == "__main__":
    main()
