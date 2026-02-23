from __future__ import annotations

import cProfile
import csv
import io
import json
import pstats
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aidan_custom.bloch_ed.haldane import build_haldane_projected_hamiltonian
from aidan_custom.bloch_ed.workflow import solve_projected_all_momentum_sectors

RESULT_DIR = Path(__file__).resolve().parent
RAW_DIR = RESULT_DIR / "raw_data"
PLOTS_DIR = RESULT_DIR / "plots"


def build_model() -> tuple[object, dict[str, float]]:
    # Written with Codex 02-21-26.
    params = {
        "Lx": 4,
        "Ly": 4,
        "n_particles": 5,
        "t1": 1.0,
        "t2": float(-1.0 / (4.0 * np.cos(0.65))),
        "phi": 0.65,
        "m": 0.0,
        "V1": 1.0,
        "n_eigs": 1,
        "sparse_threshold": 60,
        "cutoff": 1.0e-12,
        "max_sector_workers": 4,
    }
    projected = build_haldane_projected_hamiltonian(
        Lx=int(params["Lx"]),
        Ly=int(params["Ly"]),
        t1=float(params["t1"]),
        t2=float(params["t2"]),
        phi=float(params["phi"]),
        m=float(params["m"]),
        selected_bands=[0],
        V1=float(params["V1"]),
    )
    return projected, params


def run_variant(
    projected_hamiltonian,
    params: dict[str, float],
    parallel_sparse_sectors: bool,
) -> dict[tuple[int, int], float]:
    # Written with Codex 02-21-26.
    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=int(params["n_particles"]),
        cutoff=float(params["cutoff"]),
        n_eigs=int(params["n_eigs"]),
        sparse_threshold=int(params["sparse_threshold"]),
        use_sparse=True,
        parallel_sparse_sectors=bool(parallel_sparse_sectors),
        max_sector_workers=int(params["max_sector_workers"]),
    )
    out: dict[tuple[int, int], float] = {}
    for sector, value in results.items():
        eigvals = np.asarray(value["eigenvalues"], dtype=np.float64)
        out[(int(sector[0]), int(sector[1]))] = float(eigvals[0])
    return out


def benchmark_variant(
    name: str,
    projected_hamiltonian,
    params: dict[str, float],
    repeats: int,
    parallel_sparse_sectors: bool,
) -> tuple[list[dict[str, float | str]], dict[tuple[int, int], float]]:
    # Written with Codex 02-21-26.
    rows: list[dict[str, float | str]] = []
    last_result: dict[tuple[int, int], float] = {}
    for repeat in range(int(repeats)):
        t0 = time.perf_counter()
        result = run_variant(
            projected_hamiltonian=projected_hamiltonian,
            params=params,
            parallel_sparse_sectors=parallel_sparse_sectors,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "variant": name,
                "repeat": int(repeat),
                "seconds": float(elapsed),
            }
        )
        last_result = result
    return rows, last_result


def profile_variant(
    projected_hamiltonian,
    params: dict[str, float],
    parallel_sparse_sectors: bool,
    output_path: Path,
    top_n: int = 30,
) -> None:
    # Written with Codex 02-21-26.
    prof = cProfile.Profile()
    prof.enable()
    run_variant(
        projected_hamiltonian=projected_hamiltonian,
        params=params,
        parallel_sparse_sectors=parallel_sparse_sectors,
    )
    prof.disable()
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(int(top_n))
    output_path.write_text(stream.getvalue(), encoding="utf-8")


def write_timing_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    # Written with Codex 02-21-26.
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "repeat", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def plot_timings(rows: list[dict[str, float | str]], output_path: Path) -> dict[str, float]:
    # Written with Codex 02-21-26.
    variants = sorted({str(row["variant"]) for row in rows})
    means = np.asarray(
        [
            np.mean(
                [
                    float(row["seconds"])
                    for row in rows
                    if str(row["variant"]) == variant
                ]
            )
            for variant in variants
        ],
        dtype=np.float64,
    )
    stds = np.asarray(
        [
            np.std(
                [
                    float(row["seconds"])
                    for row in rows
                    if str(row["variant"]) == variant
                ],
                ddof=1,
            )
            for variant in variants
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(variants), dtype=np.int64)
    ax.bar(x, means, yerr=stds, capsize=6.0, color=["#7fb8ff", "#2f5d9a"])
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Sparse sector solve: sequential vs threaded")
    ax.grid(axis="y", alpha=0.25)

    if "sequential_sparse" in variants and "threaded_sparse" in variants:
        seq_idx = variants.index("sequential_sparse")
        thr_idx = variants.index("threaded_sparse")
        rel = float(means[thr_idx] / means[seq_idx])
        ax.text(
            float(thr_idx),
            float(means[thr_idx] + stds[thr_idx] + 0.02),
            f"threaded/seq={rel:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        variant: float(mean) for variant, mean in zip(variants, means, strict=True)
    }


def main() -> None:
    # Written with Codex 02-21-26.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    projected, params = build_model()

    run_variant(projected, params, parallel_sparse_sectors=False)
    run_variant(projected, params, parallel_sparse_sectors=True)

    repeats = 4
    seq_rows, seq_energies = benchmark_variant(
        name="sequential_sparse",
        projected_hamiltonian=projected,
        params=params,
        repeats=repeats,
        parallel_sparse_sectors=False,
    )
    thr_rows, thr_energies = benchmark_variant(
        name="threaded_sparse",
        projected_hamiltonian=projected,
        params=params,
        repeats=repeats,
        parallel_sparse_sectors=True,
    )
    rows = seq_rows + thr_rows

    shared = sorted(set(seq_energies).intersection(thr_energies))
    max_abs_energy_diff = float(
        max(abs(seq_energies[s] - thr_energies[s]) for s in shared)
        if len(shared) > 0
        else 0.0
    )

    write_timing_csv(rows=rows, output_path=RAW_DIR / "timing_raw.csv")
    means = plot_timings(rows=rows, output_path=PLOTS_DIR / "timing_comparison.png")

    seq_mean = float(means["sequential_sparse"])
    thr_mean = float(means["threaded_sparse"])
    summary = {
        "params": params,
        "repeats": int(repeats),
        "sequential_sparse_mean_seconds": seq_mean,
        "threaded_sparse_mean_seconds": thr_mean,
        "threaded_over_sequential": float(thr_mean / seq_mean),
        "sequential_over_threaded": float(seq_mean / thr_mean),
        "max_abs_energy_diff": max_abs_energy_diff,
    }
    (RAW_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    profile_variant(
        projected_hamiltonian=projected,
        params=params,
        parallel_sparse_sectors=False,
        output_path=RAW_DIR / "cprofile_sequential.txt",
    )
    profile_variant(
        projected_hamiltonian=projected,
        params=params,
        parallel_sparse_sectors=True,
        output_path=RAW_DIR / "cprofile_threaded.txt",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
