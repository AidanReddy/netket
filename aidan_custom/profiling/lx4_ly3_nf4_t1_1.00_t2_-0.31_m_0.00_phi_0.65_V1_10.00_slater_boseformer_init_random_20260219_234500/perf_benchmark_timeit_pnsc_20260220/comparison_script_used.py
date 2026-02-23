import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def find_scope_seconds(summary, scope_name):
    # Written with Codex 02-20-26.
    for scope in summary["top_level_scopes"]:
        if scope["name"] == scope_name:
            return float(scope["seconds"])
    return None


def main():
    # Written with Codex 02-20-26.
    matplotlib.use("Agg")

    benchmark_dir = Path(__file__).resolve().parent
    job_dir = benchmark_dir.parent
    baseline_summary_path = (
        job_dir / "perf_benchmark_timeit_20260220" / "raw_data" / "timing_summary.json"
    )
    conserving_summary_path = benchmark_dir / "raw_data" / "timing_summary.json"

    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    conserving = json.loads(conserving_summary_path.read_text(encoding="utf-8"))

    total_base = float(baseline["timer_total_seconds"])
    total_cons = float(conserving["timer_total_seconds"])
    fwd_base = find_scope_seconds(baseline, "VMC_SR._forward_and_backward")
    fwd_cons = find_scope_seconds(conserving, "VMC_SR._forward_and_backward")

    comparison = {
        "baseline_summary": str(baseline_summary_path),
        "conserving_summary": str(conserving_summary_path),
        "timed_n_iter_baseline": int(baseline["timed_n_iter"]),
        "timed_n_iter_conserving": int(conserving["timed_n_iter"]),
        "total_seconds_baseline": total_base,
        "total_seconds_conserving": total_cons,
        "total_seconds_delta": total_cons - total_base,
        "total_seconds_relative_percent": 100.0 * (total_cons / total_base - 1.0),
        "forward_backward_seconds_baseline": fwd_base,
        "forward_backward_seconds_conserving": fwd_cons,
        "forward_backward_delta": (
            None if (fwd_base is None or fwd_cons is None) else fwd_cons - fwd_base
        ),
        "forward_backward_relative_percent": (
            None
            if (fwd_base is None or fwd_cons is None)
            else 100.0 * (fwd_cons / fwd_base - 1.0)
        ),
    }

    (benchmark_dir / "raw_data" / "timing_comparison_vs_baseline.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = ["baseline", "conserving"]
    values = np.asarray([total_base, total_cons], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = ["tab:blue", "tab:green"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Total timed window (s)")
    ax.set_title("Benchmark comparison: total time (20 timed steps)")
    ax.grid(alpha=0.25, axis="y")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(benchmark_dir / "timing_comparison_total_seconds.png", dpi=180)
    plt.close(fig)

    (benchmark_dir / "comparison_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Saved comparison JSON to {benchmark_dir / 'raw_data' / 'timing_comparison_vs_baseline.json'}")
    print(f"Saved comparison plot to {benchmark_dir / 'timing_comparison_total_seconds.png'}")


if __name__ == "__main__":
    main()
