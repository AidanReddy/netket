import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Written with Codex 02-20-26.
    matplotlib.use("Agg")

    benchmark_dir = Path(__file__).resolve().parent
    raw_data_dir = benchmark_dir / "raw_data"
    summary_path = raw_data_dir / "timing_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    scopes = summary["top_level_scopes"]
    names = [scope["name"] for scope in scopes]
    percents = np.asarray([scope["percent_of_total"] for scope in scopes], dtype=np.float64)

    fig_height = max(3.2, 0.7 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(names))
    ax.barh(y, percents, color="tab:blue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Percent of total timed window (%)")
    ax.set_title("NetKet timeit breakdown (top-level scopes)")
    ax.grid(alpha=0.25, axis="x")

    for idx, value in enumerate(percents):
        ax.text(value + 0.1, idx, f"{value:.3f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(benchmark_dir / "timing_breakdown_top_level.png", dpi=180)
    plt.close(fig)

    (benchmark_dir / "plot_script_used.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Saved: {benchmark_dir / 'timing_breakdown_top_level.png'}")


if __name__ == "__main__":
    main()
