import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
import numpy as np

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import (
    LogBoseFormerProduct,
    LogSlaterBoseFormer,
    LogSlaterDeterminant,
    make_supercell_reciprocal_vectors_from_graph,
)


def flatten_samples(samples):
    # Written with Codex 02-20-26.
    samples_np = np.asarray(samples)
    if samples_np.ndim == 2:
        return samples_np
    if samples_np.ndim == 3:
        return samples_np.reshape((-1, samples_np.shape[-1]))
    raise ValueError(f"Unexpected sample shape: {samples_np.shape}")


def benchmark_jitted_function(jitted_fn, fn_args, n_warmup, n_repeats, trace_name):
    # Written with Codex 02-20-26.
    for _ in range(n_warmup):
        warmup_out = jitted_fn(*fn_args)
        jax.block_until_ready(warmup_out)

    times = np.empty(n_repeats, dtype=np.float64)
    for i in range(n_repeats):
        start = time.perf_counter()
        with jax.profiler.TraceAnnotation(trace_name):
            out = jitted_fn(*fn_args)
        jax.block_until_ready(out)
        times[i] = time.perf_counter() - start
    return times


def write_repeat_csv(path, timing_rows):
    # Written with Codex 02-20-26.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "component", "repeat", "seconds", "milliseconds"],
        )
        writer.writeheader()
        for row in timing_rows:
            writer.writerow(row)


def summarize_times(times):
    # Written with Codex 02-20-26.
    return float(np.mean(times)), float(np.std(times))


def compute_component_shares(slater_seconds, boseformer_seconds):
    # Written with Codex 02-20-26.
    total = slater_seconds + boseformer_seconds
    if total <= 0.0:
        return total, float("nan"), float("nan")
    return total, slater_seconds / total, boseformer_seconds / total


def plot_component_share(
    path,
    title,
    slater_share,
    boseformer_share,
    slater_mean_ms,
    boseformer_mean_ms,
):
    # Written with Codex 02-20-26.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = ["Slater", "Boseformer"]
    shares = [100.0 * slater_share, 100.0 * boseformer_share]
    colors = ["tab:blue", "tab:orange"]
    bars = ax.bar(labels, shares, color=colors, alpha=0.9)
    ax.set_ylabel("Percent of component time")
    ax.set_ylim(0.0, 100.0)
    ax.set_title(title)
    for bar, share, mean_ms in zip(
        bars, shares, [slater_mean_ms, boseformer_mean_ms], strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{share:.1f}%\n{mean_ms:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    # Written with Codex 02-20-26.
    matplotlib.use("Agg")

    devices = jax.devices()
    backend = jax.default_backend()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("devices:", devices)
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend}. Refusing to run benchmark on CPU."
        )

    benchmark_dir = Path(__file__).resolve().parent
    job_dir = benchmark_dir.parent
    raw_data_dir = benchmark_dir / "raw_data"
    plots_dir = benchmark_dir / "plots_optimization"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Match the production 5x3 MC setup.
    t1 = 1.0
    phi = 0.65
    t2 = -1.0 / (4.0 * np.cos(phi))
    m = 0.0
    Lx = 5
    Ly = 3
    V1 = 10.0
    n_fermions = 5
    n_samples = 1024 * 4
    n_discard_per_chain = 4
    n_chains = 512
    sweep_size = n_fermions

    # Model parameters from the production job.
    boseformer_num_layers = 2
    boseformer_d_model = 32
    boseformer_n_heads = 2
    boseformer_mlp_hidden_factor = 2

    # Timing controls.
    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "512"))
    n_warmup = int(os.environ.get("BENCHMARK_WARMUP", "6"))
    n_repeats = int(os.environ.get("BENCHMARK_REPEATS", "40"))

    resume_variables_path = Path(
        os.environ.get("RESUME_VSTATE_PATH", str(job_dir / "vstate_variables.mpack"))
    )
    if not resume_variables_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_variables_path}")
    print(f"Resuming from checkpoint variables at: {resume_variables_path}")

    graph, hi, _ = build_haldane_hamiltonian(
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

    sampler = nk.sampler.MetropolisFermionHop(
        hi,
        graph=graph,
        n_chains=n_chains,
        sweep_size=sweep_size,
    )

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

    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )
    vstate.variables = flax.serialization.from_bytes(
        vstate.variables, resume_variables_path.read_bytes()
    )
    params = vstate.parameters
    print("Loaded resume checkpoint into variational state.")

    sample_states = flatten_samples(vstate.samples)
    if sample_states.shape[0] < eval_batch_size:
        reps = int(np.ceil(eval_batch_size / float(sample_states.shape[0])))
        sample_states = np.tile(sample_states, (reps, 1))
    eval_states = jnp.asarray(sample_states[:eval_batch_size])
    print(f"Eval states shape: {tuple(eval_states.shape)}")

    slater_module = LogSlaterDeterminant(
        hilbert=hi,
        param_dtype=jnp.float64,
        split_complex_params=True,
        initial_m_orbitals=None,
    )
    boseformer_module = LogBoseFormerProduct(
        n_particles=hi.n_fermions,
        positions=positions_hashable,
        g_vectors=g_vectors_hashable,
        num_layers=boseformer_num_layers,
        d_model=boseformer_d_model,
        n_heads=boseformer_n_heads,
        mlp_hidden_factor=boseformer_mlp_hidden_factor,
        param_dtype=jnp.float64,
    )

    slater_params = params["slater"]
    boseformer_params = params["boseformer"]

    @jax.jit
    def eval_full(states):
        # Written with Codex 02-20-26.
        return model.apply({"params": params}, states)

    @jax.jit
    def eval_slater(states):
        # Written with Codex 02-20-26.
        return slater_module.apply({"params": slater_params}, states)

    @jax.jit
    def eval_boseformer(states):
        # Written with Codex 02-20-26.
        return boseformer_module.apply({"params": boseformer_params}, states)

    def full_loss_fn(model_params, states):
        # Written with Codex 02-20-26.
        return jnp.real(jnp.mean(model.apply({"params": model_params}, states)))

    def slater_loss_fn(slater_model_params, states):
        # Written with Codex 02-20-26.
        return jnp.real(
            jnp.mean(slater_module.apply({"params": slater_model_params}, states))
        )

    def boseformer_loss_fn(boseformer_model_params, states):
        # Written with Codex 02-20-26.
        return jnp.real(
            jnp.mean(boseformer_module.apply({"params": boseformer_model_params}, states))
        )

    full_value_and_grad = jax.jit(jax.value_and_grad(full_loss_fn))
    slater_value_and_grad = jax.jit(jax.value_and_grad(slater_loss_fn))
    boseformer_value_and_grad = jax.jit(jax.value_and_grad(boseformer_loss_fn))

    full_forward_times = benchmark_jitted_function(
        eval_full,
        (eval_states,),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_full",
    )
    slater_forward_times = benchmark_jitted_function(
        eval_slater,
        (eval_states,),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_slater",
    )
    boseformer_forward_times = benchmark_jitted_function(
        eval_boseformer,
        (eval_states,),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_boseformer",
    )

    full_forward_backward_times = benchmark_jitted_function(
        full_value_and_grad,
        (params, eval_states),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_backward_full",
    )
    slater_forward_backward_times = benchmark_jitted_function(
        slater_value_and_grad,
        (slater_params, eval_states),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_backward_slater",
    )
    boseformer_forward_backward_times = benchmark_jitted_function(
        boseformer_value_and_grad,
        (boseformer_params, eval_states),
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        trace_name="forward_backward_boseformer",
    )

    full_backward_estimate_times = full_forward_backward_times - full_forward_times
    slater_backward_estimate_times = slater_forward_backward_times - slater_forward_times
    boseformer_backward_estimate_times = (
        boseformer_forward_backward_times - boseformer_forward_times
    )

    full_forward_mean, full_forward_std = summarize_times(full_forward_times)
    slater_forward_mean, slater_forward_std = summarize_times(slater_forward_times)
    boseformer_forward_mean, boseformer_forward_std = summarize_times(
        boseformer_forward_times
    )

    (
        full_forward_backward_mean,
        full_forward_backward_std,
    ) = summarize_times(full_forward_backward_times)
    (
        slater_forward_backward_mean,
        slater_forward_backward_std,
    ) = summarize_times(slater_forward_backward_times)
    (
        boseformer_forward_backward_mean,
        boseformer_forward_backward_std,
    ) = summarize_times(boseformer_forward_backward_times)

    full_backward_estimate_mean, full_backward_estimate_std = summarize_times(
        full_backward_estimate_times
    )
    slater_backward_estimate_mean, slater_backward_estimate_std = summarize_times(
        slater_backward_estimate_times
    )
    (
        boseformer_backward_estimate_mean,
        boseformer_backward_estimate_std,
    ) = summarize_times(boseformer_backward_estimate_times)

    (
        forward_component_sum,
        slater_forward_share,
        boseformer_forward_share,
    ) = compute_component_shares(slater_forward_mean, boseformer_forward_mean)
    (
        forward_backward_component_sum,
        slater_forward_backward_share,
        boseformer_forward_backward_share,
    ) = compute_component_shares(
        slater_forward_backward_mean, boseformer_forward_backward_mean
    )
    (
        backward_component_sum,
        slater_backward_estimate_share,
        boseformer_backward_estimate_share,
    ) = compute_component_shares(
        slater_backward_estimate_mean, boseformer_backward_estimate_mean
    )

    summary = {
        "job_dir": str(job_dir),
        "benchmark_dir": str(benchmark_dir),
        "backend": backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
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
        "sampling_setup": {
            "n_samples": n_samples,
            "n_discard_per_chain": n_discard_per_chain,
            "n_chains": n_chains,
            "sweep_size": sweep_size,
            "eval_batch_size": eval_batch_size,
        },
        "timing_controls": {
            "n_warmup": n_warmup,
            "n_repeats": n_repeats,
        },
        "timing_seconds": {
            "forward": {
                "full_mean": full_forward_mean,
                "full_std": full_forward_std,
                "slater_mean": slater_forward_mean,
                "slater_std": slater_forward_std,
                "boseformer_mean": boseformer_forward_mean,
                "boseformer_std": boseformer_forward_std,
                "slater_plus_boseformer_mean": forward_component_sum,
                "full_minus_component_sum": full_forward_mean - forward_component_sum,
            },
            "forward_backward": {
                "full_mean": full_forward_backward_mean,
                "full_std": full_forward_backward_std,
                "slater_mean": slater_forward_backward_mean,
                "slater_std": slater_forward_backward_std,
                "boseformer_mean": boseformer_forward_backward_mean,
                "boseformer_std": boseformer_forward_backward_std,
                "slater_plus_boseformer_mean": forward_backward_component_sum,
                "full_minus_component_sum": (
                    full_forward_backward_mean - forward_backward_component_sum
                ),
            },
            "backward_estimate": {
                "full_mean": full_backward_estimate_mean,
                "full_std": full_backward_estimate_std,
                "slater_mean": slater_backward_estimate_mean,
                "slater_std": slater_backward_estimate_std,
                "boseformer_mean": boseformer_backward_estimate_mean,
                "boseformer_std": boseformer_backward_estimate_std,
                "slater_plus_boseformer_mean": backward_component_sum,
                "full_minus_component_sum": (
                    full_backward_estimate_mean - backward_component_sum
                ),
            },
        },
        "component_shares_percent": {
            "forward": {
                "slater": 100.0 * slater_forward_share,
                "boseformer": 100.0 * boseformer_forward_share,
            },
            "forward_backward": {
                "slater": 100.0 * slater_forward_backward_share,
                "boseformer": 100.0 * boseformer_forward_backward_share,
            },
            "backward_estimate": {
                "slater": 100.0 * slater_backward_estimate_share,
                "boseformer": 100.0 * boseformer_backward_estimate_share,
            },
        },
        "resume_variables_path": str(resume_variables_path),
    }
    (raw_data_dir / "timing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    timing_rows = []
    phase_component_times = {
        "forward": {
            "full": full_forward_times,
            "slater": slater_forward_times,
            "boseformer": boseformer_forward_times,
        },
        "forward_backward": {
            "full": full_forward_backward_times,
            "slater": slater_forward_backward_times,
            "boseformer": boseformer_forward_backward_times,
        },
        "backward_estimate": {
            "full": full_backward_estimate_times,
            "slater": slater_backward_estimate_times,
            "boseformer": boseformer_backward_estimate_times,
        },
    }
    for phase, component_map in phase_component_times.items():
        for component, times in component_map.items():
            for idx, seconds in enumerate(times):
                timing_rows.append(
                    {
                        "phase": phase,
                        "component": component,
                        "repeat": idx,
                        "seconds": f"{float(seconds):.10f}",
                        "milliseconds": f"{1000.0 * float(seconds):.6f}",
                    }
                )
    write_repeat_csv(raw_data_dir / "timing_repeats.csv", timing_rows)

    plot_component_share(
        plots_dir / "slater_vs_boseformer_time_share.png",
        f"Forward compute share (MC eval batch={eval_batch_size})",
        slater_forward_share,
        boseformer_forward_share,
        1000.0 * slater_forward_mean,
        1000.0 * boseformer_forward_mean,
    )
    plot_component_share(
        plots_dir / "slater_vs_boseformer_forward_backward_time_share.png",
        f"Forward+backward compute share (MC eval batch={eval_batch_size})",
        slater_forward_backward_share,
        boseformer_forward_backward_share,
        1000.0 * slater_forward_backward_mean,
        1000.0 * boseformer_forward_backward_mean,
    )
    plot_component_share(
        plots_dir / "slater_vs_boseformer_backward_estimate_time_share.png",
        f"Estimated backward compute share (MC eval batch={eval_batch_size})",
        slater_backward_estimate_share,
        boseformer_backward_estimate_share,
        1000.0 * slater_backward_estimate_mean,
        1000.0 * boseformer_backward_estimate_mean,
    )

    phase_labels = ["Forward", "Forward+Backward", "Backward est."]
    full_ms = [
        1000.0 * full_forward_mean,
        1000.0 * full_forward_backward_mean,
        1000.0 * full_backward_estimate_mean,
    ]
    slater_ms = [
        1000.0 * slater_forward_mean,
        1000.0 * slater_forward_backward_mean,
        1000.0 * slater_backward_estimate_mean,
    ]
    boseformer_ms = [
        1000.0 * boseformer_forward_mean,
        1000.0 * boseformer_forward_backward_mean,
        1000.0 * boseformer_backward_estimate_mean,
    ]
    x = np.arange(len(phase_labels), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width, full_ms, width, label="Full model", color="tab:green", alpha=0.9)
    ax.bar(x, slater_ms, width, label="Slater", color="tab:blue", alpha=0.9)
    ax.bar(
        x + width,
        boseformer_ms,
        width,
        label="Boseformer",
        color="tab:orange",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Mean time per call [ms]")
    ax.set_title("Timing by phase on sampled MC states")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "times_by_phase_ms.png", dpi=180)
    plt.close(fig)

    (benchmark_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(
        "Forward mean times [ms]: "
        f"full={1000.0 * full_forward_mean:.3f}, "
        f"slater={1000.0 * slater_forward_mean:.3f}, "
        f"boseformer={1000.0 * boseformer_forward_mean:.3f}"
    )
    print(
        "Forward+backward mean times [ms]: "
        f"full={1000.0 * full_forward_backward_mean:.3f}, "
        f"slater={1000.0 * slater_forward_backward_mean:.3f}, "
        f"boseformer={1000.0 * boseformer_forward_backward_mean:.3f}"
    )
    print(
        "Estimated backward mean times [ms]: "
        f"full={1000.0 * full_backward_estimate_mean:.3f}, "
        f"slater={1000.0 * slater_backward_estimate_mean:.3f}, "
        f"boseformer={1000.0 * boseformer_backward_estimate_mean:.3f}"
    )
    print(
        "Forward share [% of Slater+Boseformer]: "
        f"slater={100.0 * slater_forward_share:.2f}, "
        f"boseformer={100.0 * boseformer_forward_share:.2f}"
    )
    print(
        "Forward+backward share [% of Slater+Boseformer]: "
        f"slater={100.0 * slater_forward_backward_share:.2f}, "
        f"boseformer={100.0 * boseformer_forward_backward_share:.2f}"
    )
    print(
        "Estimated backward share [% of Slater+Boseformer]: "
        f"slater={100.0 * slater_backward_estimate_share:.2f}, "
        f"boseformer={100.0 * boseformer_backward_estimate_share:.2f}"
    )
    print(f"Saved benchmark outputs under: {benchmark_dir}")


if __name__ == "__main__":
    main()
