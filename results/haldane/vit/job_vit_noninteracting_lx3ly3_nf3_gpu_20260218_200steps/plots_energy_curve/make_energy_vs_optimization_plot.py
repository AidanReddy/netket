#!/usr/bin/env python3
"""
Create an energy-vs-optimization-step plot from a NetKet JsonLog file.
"""

from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


job_dir = Path(__file__).resolve().parents[1]
plot_dir = Path(__file__).resolve().parent

train_log_path = job_dir / "train.log"
summary_path = job_dir / "summary.json"

with train_log_path.open("r", encoding="utf-8") as f:
    train_log = json.load(f)

iters = np.asarray(train_log["Energy"]["iters"], dtype=np.int64)
energy_real = np.asarray(train_log["Energy"]["Mean"]["real"], dtype=np.float64)

ref_energy = None
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    ref_energy = summary.get("results", {}).get("ref_bloch", None)
    if ref_energy is not None:
        ref_energy = float(ref_energy)

csv_path = plot_dir / "energy_vs_optimization.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "energy_mean_real"])
    for it, ene in zip(iters.tolist(), energy_real.tolist()):
        writer.writerow([it, ene])

fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
ax.plot(iters, energy_real, linewidth=1.8, color="#1f77b4", label="VMC energy")
if ref_energy is not None:
    ax.axhline(
        ref_energy,
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label=f"Reference ({ref_energy:.6f})",
    )

ax.set_xlabel("Optimization step")
ax.set_ylabel("Energy")
ax.set_title("Energy vs Optimization Step")
ax.grid(alpha=0.3)
ax.legend(loc="best")

png_path = plot_dir / "energy_vs_optimization.png"
fig.savefig(png_path, dpi=220)
plt.close(fig)

print(f"Wrote: {png_path}")
print(f"Wrote: {csv_path}")
