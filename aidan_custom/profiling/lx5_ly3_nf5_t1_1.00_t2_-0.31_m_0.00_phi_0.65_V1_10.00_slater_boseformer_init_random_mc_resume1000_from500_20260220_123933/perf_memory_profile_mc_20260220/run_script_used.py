import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


def pytree_nbytes(tree):
    # Written with Codex 02-20-26.
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(leaf)
        total += int(arr.nbytes)
    return int(total)


def get_visible_physical_gpu_index():
    # Written with Codex 02-20-26.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    token = visible.split(",")[0].strip()
    try:
        return int(token)
    except ValueError:
        return None


def query_process_gpu_memory_mib(pid):
    # Written with Codex 02-20-26.
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    total_mib = 0.0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if row_pid == int(pid):
            total_mib += used_mib
            found = True
    return total_mib if found else None


def collect_device_memory_stats():
    # Written with Codex 02-20-26.
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return {}
    if stats is None:
        return {}
    clean = {}
    for key, value in stats.items():
        if isinstance(value, (int, float, np.number)):
            clean[str(key)] = float(value)
    return clean


def capture_memory_snapshot(label):
    # Written with Codex 02-20-26.
    return {
        "label": label,
        "time_unix_s": float(time.time()),
        "pid": int(os.getpid()),
        "process_gpu_memory_mib": query_process_gpu_memory_mib(pid=os.getpid()),
        "device_memory_stats": collect_device_memory_stats(),
    }


def read_snapshot_value(snapshots, label, key):
    # Written with Codex 02-20-26.
    for snap in snapshots:
        if snap.get("label") == label:
            return snap.get(key)
    return None


def read_snapshot_stat(snapshots, label, stat_key):
    # Written with Codex 02-20-26.
    for snap in snapshots:
        if snap.get("label") == label:
            return snap.get("device_memory_stats", {}).get(stat_key)
    return None


def safe_delta(a, b):
    # Written with Codex 02-20-26.
    if a is None or b is None:
        return None
    return float(a) - float(b)


def run_single_component(component, benchmark_dir, raw_data_dir):
    # Written with Codex 02-20-26.
    if component not in ("full", "slater", "boseformer"):
        raise ValueError(f"Unknown component: {component}")

    devices = jax.devices()
    backend = jax.default_backend()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("devices:", devices)
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend}. Refusing to run memory profile on CPU."
        )

    job_dir = benchmark_dir.parent

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

    boseformer_num_layers = 2
    boseformer_d_model = 32
    boseformer_n_heads = 2
    boseformer_mlp_hidden_factor = 2

    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "512"))

    resume_variables_path = Path(
        os.environ.get("RESUME_VSTATE_PATH", str(job_dir / "vstate_variables.mpack"))
    )
    if not resume_variables_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_variables_path}")
    print(f"Component={component} | resume from: {resume_variables_path}")

    snapshots = [capture_memory_snapshot("startup")]

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

    full_model = LogSlaterBoseFormer(
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
    slater_model = LogSlaterDeterminant(
        hilbert=hi,
        param_dtype=jnp.float64,
        split_complex_params=True,
        initial_m_orbitals=None,
    )
    boseformer_model = LogBoseFormerProduct(
        n_particles=hi.n_fermions,
        positions=positions_hashable,
        g_vectors=g_vectors_hashable,
        num_layers=boseformer_num_layers,
        d_model=boseformer_d_model,
        n_heads=boseformer_n_heads,
        mlp_hidden_factor=boseformer_mlp_hidden_factor,
        param_dtype=jnp.float64,
    )

    vstate = nk.vqs.MCState(
        sampler,
        full_model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )
    vstate.variables = flax.serialization.from_bytes(
        vstate.variables, resume_variables_path.read_bytes()
    )
    params_full = vstate.parameters
    params_slater = params_full["slater"]
    params_boseformer = params_full["boseformer"]
    param_bytes = {
        "full": pytree_nbytes(params_full),
        "slater": pytree_nbytes(params_slater),
        "boseformer": pytree_nbytes(params_boseformer),
    }

    sample_states = flatten_samples(vstate.samples)
    if sample_states.shape[0] < eval_batch_size:
        reps = int(np.ceil(eval_batch_size / float(sample_states.shape[0])))
        sample_states = np.tile(sample_states, (reps, 1))
    eval_states = jnp.asarray(sample_states[:eval_batch_size])

    snapshots.append(capture_memory_snapshot("model_ready"))

    if component == "full":
        component_params = params_full

        def forward_fn(p, states):
            # Written with Codex 02-20-26.
            return full_model.apply({"params": p}, states)

        def loss_fn(p, states):
            # Written with Codex 02-20-26.
            return jnp.real(jnp.mean(full_model.apply({"params": p}, states)))

    elif component == "slater":
        component_params = params_slater

        def forward_fn(p, states):
            # Written with Codex 02-20-26.
            return slater_model.apply({"params": p}, states)

        def loss_fn(p, states):
            # Written with Codex 02-20-26.
            return jnp.real(jnp.mean(slater_model.apply({"params": p}, states)))

    else:
        component_params = params_boseformer

        def forward_fn(p, states):
            # Written with Codex 02-20-26.
            return boseformer_model.apply({"params": p}, states)

        def loss_fn(p, states):
            # Written with Codex 02-20-26.
            return jnp.real(jnp.mean(boseformer_model.apply({"params": p}, states)))

    forward_jit = jax.jit(forward_fn)
    value_and_grad_jit = jax.jit(jax.value_and_grad(loss_fn))

    forward_out = forward_jit(component_params, eval_states)
    jax.block_until_ready(forward_out)
    snapshots.append(capture_memory_snapshot("after_forward_compile"))

    forward_out = forward_jit(component_params, eval_states)
    jax.block_until_ready(forward_out)
    snapshots.append(capture_memory_snapshot("after_forward_steady"))

    loss_value, grads = value_and_grad_jit(component_params, eval_states)
    jax.block_until_ready(loss_value)
    jax.tree_util.tree_map(jax.block_until_ready, grads)
    snapshots.append(
        capture_memory_snapshot("after_forward_backward_compile")
    )

    loss_value, grads = value_and_grad_jit(component_params, eval_states)
    jax.block_until_ready(loss_value)
    jax.tree_util.tree_map(jax.block_until_ready, grads)
    snapshots.append(
        capture_memory_snapshot("after_forward_backward_steady")
    )

    memory_profile_file = raw_data_dir / f"device_memory_{component}_after_fwd_bwd.prof"
    try:
        jax.profiler.save_device_memory_profile(str(memory_profile_file))
        memory_profile_saved = True
    except Exception:
        memory_profile_saved = False

    model_ready_mib = read_snapshot_value(
        snapshots, "model_ready", "process_gpu_memory_mib"
    )
    fwd_steady_mib = read_snapshot_value(
        snapshots, "after_forward_steady", "process_gpu_memory_mib"
    )
    fwd_bwd_steady_mib = read_snapshot_value(
        snapshots, "after_forward_backward_steady", "process_gpu_memory_mib"
    )
    process_peak_mib = None
    process_mibs = [
        snap.get("process_gpu_memory_mib")
        for snap in snapshots
        if snap.get("process_gpu_memory_mib") is not None
    ]
    if len(process_mibs) > 0:
        process_peak_mib = float(np.max(np.asarray(process_mibs, dtype=np.float64)))

    peak_bytes_in_use = read_snapshot_stat(
        snapshots, "after_forward_backward_steady", "peak_bytes_in_use"
    )
    bytes_in_use = read_snapshot_stat(
        snapshots, "after_forward_backward_steady", "bytes_in_use"
    )
    bytes_limit = read_snapshot_stat(
        snapshots, "after_forward_backward_steady", "bytes_limit"
    )

    result = {
        "component": component,
        "backend": backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "physical_gpu_index": get_visible_physical_gpu_index(),
        "pid": int(os.getpid()),
        "resume_variables_path": str(resume_variables_path),
        "eval_batch_size": eval_batch_size,
        "param_bytes": {
            "used_component": int(param_bytes[component]),
            "full": int(param_bytes["full"]),
            "slater": int(param_bytes["slater"]),
            "boseformer": int(param_bytes["boseformer"]),
        },
        "process_gpu_memory_mib": {
            "model_ready": model_ready_mib,
            "after_forward_steady": fwd_steady_mib,
            "after_forward_backward_steady": fwd_bwd_steady_mib,
            "delta_forward_minus_model_ready": safe_delta(fwd_steady_mib, model_ready_mib),
            "delta_fwd_bwd_minus_model_ready": safe_delta(
                fwd_bwd_steady_mib, model_ready_mib
            ),
            "peak_over_snapshots": process_peak_mib,
        },
        "device_memory_stats_after_forward_backward_steady": {
            "bytes_in_use": bytes_in_use,
            "peak_bytes_in_use": peak_bytes_in_use,
            "bytes_limit": bytes_limit,
        },
        "memory_profile_file": str(memory_profile_file),
        "memory_profile_saved": bool(memory_profile_saved),
        "snapshots": snapshots,
    }

    out_path = raw_data_dir / f"component_{component}_memory_profile.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{component}: process peak={process_peak_mib} MiB, "
        f"delta_fwd_bwd={result['process_gpu_memory_mib']['delta_fwd_bwd_minus_model_ready']} MiB"
    )
    print(f"Wrote {out_path}")


def make_bar_plot(path, labels, values, ylabel, title, color):
    # Written with Codex 02-20-26.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, values, color=color, alpha=0.9)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_all_components(benchmark_dir, raw_data_dir, plots_dir):
    # Written with Codex 02-20-26.
    components = ("full", "slater", "boseformer")
    for component in components:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--component", component]
        env = os.environ.copy()
        subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=env)

    per_component = {}
    for component in components:
        path = raw_data_dir / f"component_{component}_memory_profile.json"
        per_component[component] = json.loads(path.read_text(encoding="utf-8"))

    param_full_bytes = float(per_component["full"]["param_bytes"]["full"])
    param_slater_bytes = float(per_component["slater"]["param_bytes"]["used_component"])
    param_boseformer_bytes = float(
        per_component["boseformer"]["param_bytes"]["used_component"]
    )
    param_component_sum = param_slater_bytes + param_boseformer_bytes

    def get_metric(component_name, key):
        # Written with Codex 02-20-26.
        return per_component[component_name]["process_gpu_memory_mib"].get(key)

    full_peak = get_metric("full", "peak_over_snapshots")
    slater_peak = get_metric("slater", "peak_over_snapshots")
    boseformer_peak = get_metric("boseformer", "peak_over_snapshots")

    full_delta = get_metric("full", "delta_fwd_bwd_minus_model_ready")
    slater_delta = get_metric("slater", "delta_fwd_bwd_minus_model_ready")
    boseformer_delta = get_metric("boseformer", "delta_fwd_bwd_minus_model_ready")

    param_slater_share = (
        100.0 * param_slater_bytes / param_component_sum
        if param_component_sum > 0.0
        else None
    )
    param_boseformer_share = (
        100.0 * param_boseformer_bytes / param_component_sum
        if param_component_sum > 0.0
        else None
    )
    delta_component_sum = None
    delta_slater_share = None
    delta_boseformer_share = None
    if slater_delta is not None and boseformer_delta is not None:
        delta_component_sum = float(slater_delta) + float(boseformer_delta)
        if delta_component_sum > 0.0:
            delta_slater_share = 100.0 * float(slater_delta) / delta_component_sum
            delta_boseformer_share = 100.0 * float(boseformer_delta) / delta_component_sum

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "raw_data_dir": str(raw_data_dir),
        "plots_dir": str(plots_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "components": per_component,
        "aggregate": {
            "param_bytes": {
                "full": param_full_bytes,
                "slater": param_slater_bytes,
                "boseformer": param_boseformer_bytes,
                "slater_plus_boseformer": param_component_sum,
                "slater_share_percent_of_components": param_slater_share,
                "boseformer_share_percent_of_components": param_boseformer_share,
            },
            "process_peak_memory_mib": {
                "full": full_peak,
                "slater": slater_peak,
                "boseformer": boseformer_peak,
            },
            "process_delta_fwd_bwd_minus_model_ready_mib": {
                "full": full_delta,
                "slater": slater_delta,
                "boseformer": boseformer_delta,
                "slater_plus_boseformer": delta_component_sum,
                "slater_share_percent_of_components": delta_slater_share,
                "boseformer_share_percent_of_components": delta_boseformer_share,
            },
        },
    }

    (raw_data_dir / "memory_profile_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    make_bar_plot(
        plots_dir / "parameter_memory_mb.png",
        labels=["Full", "Slater", "Boseformer"],
        values=[
            param_full_bytes / (1024.0**2),
            param_slater_bytes / (1024.0**2),
            param_boseformer_bytes / (1024.0**2),
        ],
        ylabel="Parameter memory [MiB]",
        title="Parameter memory footprint",
        color=["tab:green", "tab:blue", "tab:orange"],
    )

    peak_values = [
        0.0 if full_peak is None else float(full_peak),
        0.0 if slater_peak is None else float(slater_peak),
        0.0 if boseformer_peak is None else float(boseformer_peak),
    ]
    make_bar_plot(
        plots_dir / "process_peak_memory_mib.png",
        labels=["Full", "Slater", "Boseformer"],
        values=peak_values,
        ylabel="Process GPU memory [MiB]",
        title="Peak process GPU memory over snapshots",
        color=["tab:green", "tab:blue", "tab:orange"],
    )

    delta_values = [
        0.0 if full_delta is None else float(full_delta),
        0.0 if slater_delta is None else float(slater_delta),
        0.0 if boseformer_delta is None else float(boseformer_delta),
    ]
    make_bar_plot(
        plots_dir / "process_delta_fwd_bwd_minus_model_ready_mib.png",
        labels=["Full", "Slater", "Boseformer"],
        values=delta_values,
        ylabel="GPU memory delta [MiB]",
        title="Steady fwd+bwd minus model-ready process memory",
        color=["tab:green", "tab:blue", "tab:orange"],
    )

    (benchmark_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(
        "Parameter shares (Slater vs Boseformer): "
        f"{param_slater_share:.2f}% vs {param_boseformer_share:.2f}%"
        if param_slater_share is not None and param_boseformer_share is not None
        else "Parameter shares unavailable."
    )
    if delta_slater_share is not None and delta_boseformer_share is not None:
        print(
            "Process delta shares (Slater vs Boseformer): "
            f"{delta_slater_share:.2f}% vs {delta_boseformer_share:.2f}%"
        )
    else:
        print("Process delta shares unavailable.")
    print(f"Saved memory profile outputs under: {benchmark_dir}")


def parse_args():
    # Written with Codex 02-20-26.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=["all", "full", "slater", "boseformer"],
        default="all",
    )
    return parser.parse_args()


def main():
    # Written with Codex 02-20-26.
    matplotlib.use("Agg")
    args = parse_args()
    benchmark_dir = Path(__file__).resolve().parent
    raw_data_dir = benchmark_dir / "raw_data"
    plots_dir = benchmark_dir / "plots_optimization"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.component == "all":
        run_all_components(benchmark_dir, raw_data_dir, plots_dir)
    else:
        run_single_component(args.component, benchmark_dir, raw_data_dir)


if __name__ == "__main__":
    main()
