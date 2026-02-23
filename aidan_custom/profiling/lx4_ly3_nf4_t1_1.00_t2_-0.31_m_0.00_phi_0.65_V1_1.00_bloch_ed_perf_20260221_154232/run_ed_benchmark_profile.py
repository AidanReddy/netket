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
from aidan_custom.bloch_ed.workflow import (
    solve_projected_all_momentum_sectors,
    solve_projected_sector,
)
from aidan_custom.bloch_ed.many_body import build_fock_basis

RESULT_DIR = Path(__file__).resolve().parent
RAW_DIR = RESULT_DIR / "raw_data"
PLOTS_DIR = RESULT_DIR / "plots"


def build_model() -> tuple[object, dict[str, float]]:
    # Written with Codex 02-21-26.
    params = {
        "Lx": 4,
        "Ly": 3,
        "n_particles": 4,
        "t1": 1.0,
        "t2": float(-1.0 / (4.0 * np.cos(0.65))),
        "phi": 0.65,
        "m": 0.0,
        "V1": 1.0,
        "n_eigs": 1,
        "sparse_threshold": 200,
        "cutoff": 1.0e-12,
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


def run_legacy_sector_loop(projected_hamiltonian, params: dict[str, float]) -> dict[tuple[int, int], float]:
    # Written with Codex 02-21-26.
    out: dict[tuple[int, int], float] = {}
    Lx = int(params["Lx"])
    Ly = int(params["Ly"])
    sparse_threshold = int(params["sparse_threshold"])
    for kx in range(Lx):
        for ky in range(Ly):
            basis_preview = build_fock_basis(
                n_orbitals=projected_hamiltonian.one_body.shape[0],
                n_particles=int(params["n_particles"]),
                orbital_momenta=projected_hamiltonian.orbital_momenta,
                lattice_shape=(Lx, Ly),
                momentum_sector=(kx, ky),
            )
            dim = len(basis_preview.states)
            if dim == 0:
                continue
            use_sparse_sector = dim > sparse_threshold
            basis, _, eigvals, _ = solve_projected_sector(
                projected_hamiltonian=projected_hamiltonian,
                n_particles=int(params["n_particles"]),
                momentum_sector=(kx, ky),
                cutoff=float(params["cutoff"]),
                n_eigs=int(params["n_eigs"]),
                sparse_threshold=sparse_threshold,
                use_sparse=use_sparse_sector,
            )
            out[(int(kx), int(ky))] = float(np.real(eigvals[0]))
    return out


def run_optimized_all_sectors(projected_hamiltonian, params: dict[str, float]) -> dict[tuple[int, int], float]:
    # Written with Codex 02-21-26.
    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=int(params["n_particles"]),
        cutoff=float(params["cutoff"]),
        n_eigs=int(params["n_eigs"]),
        sparse_threshold=int(params["sparse_threshold"]),
        use_sparse=True,
    )
    out: dict[tuple[int, int], float] = {}
    for sector, value in results.items():
        eigvals = np.asarray(value["eigenvalues"], dtype=np.float64)
        out[(int(sector[0]), int(sector[1]))] = float(eigvals[0])
    return out


def benchmark_variant(
    name: str,
    run_fn,
    projected_hamiltonian,
    params: dict[str, float],
    repeats: int,
) -> tuple[list[dict[str, float | str]], dict[tuple[int, int], float]]:
    # Written with Codex 02-21-26.
    rows: list[dict[str, float | str]] = []
    last_result: dict[tuple[int, int], float] = {}
    for repeat in range(int(repeats)):
        t0 = time.perf_counter()
        result = run_fn(projected_hamiltonian, params)
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
    run_fn,
    projected_hamiltonian,
    params: dict[str, float],
    output_path: Path,
    top_n: int = 30,
) -> None:
    # Written with Codex 02-21-26.
    prof = cProfile.Profile()
    prof.enable()
    run_fn(projected_hamiltonian, params)
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

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(variants), dtype=np.int64)
    ax.bar(x, means, yerr=stds, capsize=6.0, color=["#7fb8ff", "#2f5d9a"])
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("ED timing benchmark: legacy vs optimized")
    ax.grid(axis="y", alpha=0.25)

    if "legacy_loop" in variants and "optimized_all_sectors" in variants:
        legacy_idx = variants.index("legacy_loop")
        opt_idx = variants.index("optimized_all_sectors")
        speedup = float(means[legacy_idx] / means[opt_idx])
        ax.text(
            float(opt_idx),
            float(means[opt_idx] + stds[opt_idx] + 0.01),
            f"{speedup:.2f}x faster",
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

    run_legacy_sector_loop(projected, params)
    run_optimized_all_sectors(projected, params)

    repeats = 8
    legacy_rows, legacy_energies = benchmark_variant(
        "legacy_loop",
        run_legacy_sector_loop,
        projected,
        params,
        repeats=repeats,
    )
    optimized_rows, optimized_energies = benchmark_variant(
        "optimized_all_sectors",
        run_optimized_all_sectors,
        projected,
        params,
        repeats=repeats,
    )
    rows = legacy_rows + optimized_rows

    shared_sectors = sorted(set(legacy_energies).intersection(optimized_energies))
    max_abs_energy_diff = float(
        max(
            abs(legacy_energies[s] - optimized_energies[s]) for s in shared_sectors
        )
        if len(shared_sectors) > 0
        else 0.0
    )

    write_timing_csv(rows, RAW_DIR / "timing_raw.csv")
    means = plot_timings(rows, PLOTS_DIR / "timing_comparison.png")

    legacy_mean = float(means["legacy_loop"])
    optimized_mean = float(means["optimized_all_sectors"])
    summary = {
        "params": params,
        "repeats": int(repeats),
        "legacy_mean_seconds": legacy_mean,
        "optimized_mean_seconds": optimized_mean,
        "speedup": float(legacy_mean / optimized_mean),
        "max_abs_energy_diff": max_abs_energy_diff,
    }
    (RAW_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    profile_variant(
        run_legacy_sector_loop,
        projected,
        params,
        RAW_DIR / "cprofile_legacy.txt",
        top_n=30,
    )
    profile_variant(
        run_optimized_all_sectors,
        projected,
        params,
        RAW_DIR / "cprofile_optimized.txt",
        top_n=30,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
