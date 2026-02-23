import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
import numpy as np

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import (
    LogSlaterBoseFormer,
    make_supercell_reciprocal_vectors_from_graph,
)
from aidan_custom.optimization import log_optimization_diagnostics


def pytree_nbytes(tree):
    # Written with Codex 02-21-26.
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        total += int(np.asarray(leaf).nbytes)
    return int(total)


def query_process_gpu_memory_mib(pid):
    # Written with Codex 02-21-26.
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


def mib_to_gib(value):
    # Written with Codex 02-21-26.
    if value is None:
        return None
    return float(value) / 1024.0


def collect_device_memory_stats():
    # Written with Codex 02-21-26.
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


def parse_energy_mean(log_data):
    # Written with Codex 02-21-26.
    if not isinstance(log_data, dict):
        return None
    energy_data = log_data.get("Energy", None)
    if isinstance(energy_data, dict):
        mean_entry = energy_data.get("Mean", None)
        if mean_entry is not None:
            try:
                return float(np.real(np.asarray(mean_entry)))
            except Exception:
                return None
    return None


def capture_snapshot(label, step=None, log_data=None):
    # Written with Codex 02-21-26.
    process_mib = query_process_gpu_memory_mib(pid=os.getpid())
    return {
        "label": label,
        "time_unix_s": float(time.time()),
        "pid": int(os.getpid()),
        "step": None if step is None else int(step),
        "energy_mean_real": parse_energy_mean(log_data),
        "process_gpu_memory_gib": mib_to_gib(process_mib),
        "device_memory_stats": collect_device_memory_stats(),
    }


def memory_poll_worker(stop_event, interval_s, series):
    # Written with Codex 02-21-26.
    pid = int(os.getpid())
    while not stop_event.is_set():
        process_mib = query_process_gpu_memory_mib(pid=pid)
        series.append(
            {
                "time_unix_s": float(time.time()),
                "process_gpu_memory_gib": mib_to_gib(process_mib),
            }
        )
        stop_event.wait(interval_s)


def write_timeseries_csv(path, series, t0):
    # Written with Codex 02-21-26.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_since_run_start_s",
                "time_unix_s",
                "process_gpu_memory_gib",
            ],
        )
        writer.writeheader()
        for row in series:
            writer.writerow(
                {
                    "t_since_run_start_s": (
                        "" if t0 is None else f"{row['time_unix_s'] - t0:.6f}"
                    ),
                    "time_unix_s": f"{row['time_unix_s']:.6f}",
                    "process_gpu_memory_gib": (
                        "" if row["process_gpu_memory_gib"] is None else row["process_gpu_memory_gib"]
                    ),
                }
            )


def write_snapshots_csv(path, snapshots):
    # Written with Codex 02-21-26.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "step",
                "time_unix_s",
                "process_gpu_memory_gib",
                "bytes_in_use",
                "peak_bytes_in_use",
                "bytes_limit",
                "num_allocs",
                "energy_mean_real",
            ],
        )
        writer.writeheader()
        for snap in snapshots:
            stats = snap.get("device_memory_stats", {})
            writer.writerow(
                {
                    "label": snap.get("label"),
                    "step": snap.get("step"),
                    "time_unix_s": f"{snap['time_unix_s']:.6f}",
                    "process_gpu_memory_gib": snap.get("process_gpu_memory_gib"),
                    "bytes_in_use": stats.get("bytes_in_use"),
                    "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                    "bytes_limit": stats.get("bytes_limit"),
                    "num_allocs": stats.get("num_allocs"),
                    "energy_mean_real": snap.get("energy_mean_real"),
                }
            )


def make_process_memory_timeline_plot(path, series, run_start_time, run_end_time):
    # Written with Codex 02-21-26.
    xs = []
    ys = []
    for row in series:
        if row["process_gpu_memory_gib"] is None:
            continue
        if run_start_time is None:
            continue
        xs.append(row["time_unix_s"] - run_start_time)
        ys.append(row["process_gpu_memory_gib"])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if len(xs) > 0:
        ax.plot(xs, ys, lw=1.6, color="tab:blue")
    else:
        ax.text(0.5, 0.5, "No process memory samples", ha="center", va="center", transform=ax.transAxes)
    if run_start_time is not None and run_end_time is not None:
        ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(
            run_end_time - run_start_time,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
    ax.set_xlabel("Seconds since optimization run start")
    ax.set_ylabel("Process GPU memory [GiB]")
    ax.set_title("PID GPU memory timeline during full optimization")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_stage_memory_plot(path, snapshots):
    # Written with Codex 02-21-26.
    labels = [snap["label"] for snap in snapshots]
    values = [
        np.nan if snap.get("process_gpu_memory_gib") is None else snap["process_gpu_memory_gib"]
        for snap in snapshots
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    ax.plot(x, values, marker="o", lw=1.5, color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Process GPU memory [GiB]")
    ax.set_title("GPU memory snapshots by optimization stage")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    # Written with Codex 02-21-26.
    matplotlib.use("Agg")

    devices = jax.devices()
    backend = jax.default_backend()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("devices:", devices)
    print("default_backend:", backend)
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got {backend}. Refusing to run optimization memory profile on CPU."
        )

    benchmark_dir = Path(__file__).resolve().parent
    job_dir = benchmark_dir.parent
    raw_data_dir = benchmark_dir / "raw_data"
    plots_dir = benchmark_dir / "plots_optimization"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    t1 = 1.0
    phi = 0.65
    t2 = -1.0 / (4.0 * np.cos(phi))
    m = 0.0
    Lx = 5
    Ly = 3
    V1 = 10.0
    n_fermions = 5

    n_iter = int(os.environ.get("N_ITER", "3"))
    n_samples = int(os.environ.get("N_SAMPLES", str(1024 * 4)))
    n_discard_per_chain = 4
    sweep_size = n_fermions
    n_chains = 512
    learning_rate = 0.005

    boseformer_num_layers = 2
    boseformer_d_model = 32
    boseformer_n_heads = 2
    boseformer_mlp_hidden_factor = 2

    poll_interval_s = float(os.environ.get("MEM_POLL_INTERVAL_S", "0.2"))
    job_name = benchmark_dir.parent.name
    default_resume_path = job_dir / "vstate_variables.mpack"
    if not default_resume_path.exists():
        default_resume_path = (
            Path.cwd() / "results" / "boseformer" / job_name / "vstate_variables.mpack"
        )
    resume_variables_path = Path(
        os.environ.get("RESUME_VSTATE_PATH", str(default_resume_path))
    )
    if not resume_variables_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_variables_path}")
    print(f"Resuming from checkpoint variables at: {resume_variables_path}")

    snapshots = [capture_snapshot("startup")]

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
    snapshots.append(capture_snapshot("after_hamiltonian"))

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
    snapshots.append(capture_snapshot("after_model"))

    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )
    snapshots.append(capture_snapshot("after_vstate_init"))

    vstate.variables = flax.serialization.from_bytes(
        vstate.variables, resume_variables_path.read_bytes()
    )
    snapshots.append(capture_snapshot("after_resume_load"))

    param_bytes_total = pytree_nbytes(vstate.parameters)
    param_bytes_slater = pytree_nbytes(vstate.parameters["slater"])
    param_bytes_boseformer = pytree_nbytes(vstate.parameters["boseformer"])

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        ham,
        optimizer,
        variational_state=vstate,
        diag_shift=0.01,
        mode="complex",
    )
    snapshots.append(capture_snapshot("after_driver_init"))

    runtime_log = nk.logging.RuntimeLog()
    per_step_snapshots = []

    def callback(step: int, log_data: dict, driver_obj) -> bool:
        # Written with Codex 02-21-26.
        keep_going = log_optimization_diagnostics(step, log_data, driver_obj)
        snap = capture_snapshot(f"after_step_{int(step)}", step=step, log_data=log_data)
        snapshots.append(snap)
        per_step_snapshots.append(snap)
        return bool(keep_going)

    poll_series = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=memory_poll_worker,
        args=(stop_event, poll_interval_s, poll_series),
        daemon=True,
    )

    run_start_time = float(time.time())
    snapshots.append(capture_snapshot("before_run"))
    poll_thread.start()
    run_failed = False
    run_error_message = None
    try:
        driver.run(n_iter=n_iter, out=runtime_log, callback=callback, show_progress=False)
    except Exception as exc:
        run_failed = True
        run_error_message = repr(exc)
        raise
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
    run_end_time = float(time.time())
    snapshots.append(capture_snapshot("after_run"))

    try:
        jax.profiler.save_device_memory_profile(
            str(raw_data_dir / "device_memory_after_full_optimization.prof")
        )
        memory_profile_saved = True
    except Exception:
        memory_profile_saved = False

    runtime_log.serialize(raw_data_dir / "runtime_log_opt3")

    process_gibs = [
        float(snap["process_gpu_memory_gib"])
        for snap in snapshots
        if snap.get("process_gpu_memory_gib") is not None
    ]
    poll_gibs = [
        float(row["process_gpu_memory_gib"])
        for row in poll_series
        if row.get("process_gpu_memory_gib") is not None
    ]
    peak_snapshot_gib = max(process_gibs) if len(process_gibs) > 0 else None
    peak_polled_gib = max(poll_gibs) if len(poll_gibs) > 0 else None
    mean_polled_gib = float(np.mean(np.asarray(poll_gibs, dtype=np.float64))) if len(poll_gibs) > 0 else None
    overall_peak_gib = None
    if peak_snapshot_gib is not None and peak_polled_gib is not None:
        overall_peak_gib = max(peak_snapshot_gib, peak_polled_gib)
    elif peak_snapshot_gib is not None:
        overall_peak_gib = peak_snapshot_gib
    elif peak_polled_gib is not None:
        overall_peak_gib = peak_polled_gib

    model_ready_gib = None
    after_run_gib = None
    for snap in snapshots:
        if snap["label"] == "after_driver_init":
            model_ready_gib = snap.get("process_gpu_memory_gib")
        if snap["label"] == "after_run":
            after_run_gib = snap.get("process_gpu_memory_gib")

    peak_bytes_in_use = None
    for snap in snapshots:
        candidate = snap.get("device_memory_stats", {}).get("peak_bytes_in_use")
        if candidate is None:
            continue
        if peak_bytes_in_use is None or candidate > peak_bytes_in_use:
            peak_bytes_in_use = candidate

    summary = {
        "job_dir": str(job_dir),
        "benchmark_dir": str(benchmark_dir),
        "backend": backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device_kind": getattr(devices[0], "device_kind", "<unknown>"),
        "resume_variables_path": str(resume_variables_path),
        "optimization": {
            "n_iter": n_iter,
            "n_samples": n_samples,
            "n_discard_per_chain": n_discard_per_chain,
            "n_chains": n_chains,
            "sweep_size": sweep_size,
            "learning_rate": learning_rate,
            "diag_shift": 0.01,
            "mode": "complex",
        },
        "parameter_bytes": {
            "total": param_bytes_total,
            "slater": param_bytes_slater,
            "boseformer": param_bytes_boseformer,
            "slater_share_percent": (
                100.0 * param_bytes_slater / param_bytes_total
                if param_bytes_total > 0
                else None
            ),
            "boseformer_share_percent": (
                100.0 * param_bytes_boseformer / param_bytes_total
                if param_bytes_total > 0
                else None
            ),
        },
        "parameter_memory_gib": {
            "total": float(param_bytes_total) / (1024.0**3),
            "slater": float(param_bytes_slater) / (1024.0**3),
            "boseformer": float(param_bytes_boseformer) / (1024.0**3),
        },
        "process_gpu_memory_gib": {
            "model_ready": model_ready_gib,
            "after_run": after_run_gib,
            "peak_snapshot": peak_snapshot_gib,
            "peak_polled": peak_polled_gib,
            "mean_polled": mean_polled_gib,
            "overall_peak": overall_peak_gib,
            "delta_after_run_minus_model_ready": (
                None
                if model_ready_gib is None or after_run_gib is None
                else float(after_run_gib) - float(model_ready_gib)
            ),
            "delta_peak_minus_model_ready": (
                None
                if model_ready_gib is None or overall_peak_gib is None
                else float(overall_peak_gib) - float(model_ready_gib)
            ),
        },
        "device_memory_stats": {
            "peak_peak_bytes_in_use": peak_bytes_in_use,
        },
        "timing": {
            "run_start_time_unix_s": run_start_time,
            "run_end_time_unix_s": run_end_time,
            "run_duration_s": run_end_time - run_start_time,
            "poll_interval_s": poll_interval_s,
            "n_poll_samples": len(poll_series),
        },
        "run_failed": run_failed,
        "run_error_message": run_error_message,
        "memory_profile_saved": memory_profile_saved,
    }

    (raw_data_dir / "memory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (raw_data_dir / "memory_snapshots.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
    )
    (raw_data_dir / "poll_timeseries.json").write_text(
        json.dumps(poll_series, indent=2) + "\n", encoding="utf-8"
    )
    write_timeseries_csv(raw_data_dir / "process_memory_timeseries.csv", poll_series, run_start_time)
    write_snapshots_csv(raw_data_dir / "memory_snapshots.csv", snapshots)

    make_process_memory_timeline_plot(
        plots_dir / "process_memory_timeline_gib.png",
        poll_series,
        run_start_time,
        run_end_time,
    )
    make_stage_memory_plot(plots_dir / "process_memory_by_stage_gib.png", snapshots)

    (benchmark_dir / "run_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"Device kind: {summary['device_kind']}")
    print(
        "Process memory [GiB]: "
        f"model_ready={model_ready_gib}, "
        f"mean={mean_polled_gib}, "
        f"peak={overall_peak_gib}, "
        f"after_run={after_run_gib}"
    )
    print(
        "Parameter memory [GiB]: "
        f"total={param_bytes_total / (1024.0**3):.9f}, "
        f"slater={param_bytes_slater / (1024.0**3):.9f}, "
        f"boseformer={param_bytes_boseformer / (1024.0**3):.9f}"
    )
    print(f"Saved profiling outputs under: {benchmark_dir}")


if __name__ == "__main__":
    main()
