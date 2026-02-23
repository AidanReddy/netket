#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aidan_custom.geometry import (
    ANNN_VECTORS,
    DELTA_VECTORS,
    PRIMITIVE_A1,
    PRIMITIVE_A2,
    reciprocal_vectors,
    sample_shortest_representative_bz,
)


def parse_args() -> argparse.Namespace:
    # Written with Codex 02-19-26.
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Haldane-model single-particle parameters (t2, phi) at fixed t1 and m, "
            "then compute width/gap and trace-condition diagnostics."
        )
    )
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.0)
    parser.add_argument("--t2-min", type=float, default=-0.60)
    parser.add_argument("--t2-max", type=float, default=0.60)
    parser.add_argument("--n-t2", type=int, default=121)
    parser.add_argument("--phi-min", type=float, default=0.0)
    parser.add_argument("--phi-max", type=float, default=float(np.pi))
    parser.add_argument("--n-phi", type=int, default=121)
    parser.add_argument("--lx-bz-plot", type=int, default=20)
    parser.add_argument("--gap-tol", type=float, default=1.0e-8)
    parser.add_argument("--chern-target", type=float, default=1.0)
    parser.add_argument("--chern-tol", type=float, default=0.25)
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Job directory containing scripts/, raw_data/, and plots/.",
    )
    return parser.parse_args()


def precompute_k_tables(k_points: np.ndarray) -> dict[str, np.ndarray]:
    # Written with Codex 02-19-26.
    delta = np.asarray(DELTA_VECTORS, dtype=np.float64)
    annn = np.asarray(ANNN_VECTORS, dtype=np.float64)

    kd = k_points @ delta.T
    ka = k_points @ annn.T

    phase_nn = np.exp(1j * kd)
    f = -np.sum(phase_nn, axis=1)
    df_dkx = -np.sum(1j * delta[None, :, 0] * phase_nn, axis=1)
    df_dky = -np.sum(1j * delta[None, :, 1] * phase_nn, axis=1)

    cos_ka = np.cos(ka)
    sin_ka = np.sin(ka)

    sum_cos = np.sum(cos_ka, axis=1)
    sum_sin = np.sum(sin_ka, axis=1)
    sum_ax_cos = np.sum(annn[None, :, 0] * cos_ka, axis=1)
    sum_ay_cos = np.sum(annn[None, :, 1] * cos_ka, axis=1)

    return {
        "dx": np.real(f),
        "dy": -np.imag(f),
        "ddx_x": np.real(df_dkx),
        "ddx_y": -np.imag(df_dkx),
        "ddy_x": np.real(df_dky),
        "ddy_y": -np.imag(df_dky),
        "sum_cos": sum_cos,
        "sum_sin": sum_sin,
        "sum_ax_cos": sum_ax_cos,
        "sum_ay_cos": sum_ay_cos,
    }


def evaluate_metrics_at_parameters(
    t2: float,
    phi: float,
    m: float,
    tables: dict[str, np.ndarray],
    prefactor: float,
    gap_tol: float,
) -> tuple[float, float, float, float, float, float]:
    # Written with Codex 02-19-26.
    coef_sin = 2.0 * t2 * np.sin(phi)
    coef_cos = -2.0 * t2 * np.cos(phi)

    dz = m + coef_sin * tables["sum_sin"]
    d0 = coef_cos * tables["sum_cos"]

    dx = tables["dx"]
    dy = tables["dy"]
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    min_norm = float(np.min(norm))
    band_gap = 2.0 * min_norm

    if min_norm <= gap_tol:
        return np.nan, band_gap, np.inf, np.nan, np.nan, np.nan

    lower_band = d0 - norm
    band_width = float(np.max(lower_band) - np.min(lower_band))
    width_gap_ratio = band_width / band_gap

    ddx_z = coef_sin * tables["sum_ax_cos"]
    ddy_z = coef_sin * tables["sum_ay_cos"]

    ddx_x = tables["ddx_x"]
    ddx_y = tables["ddx_y"]
    ddy_x = tables["ddy_x"]
    ddy_y = tables["ddy_y"]

    inv_norm = 1.0 / norm
    inv_norm3 = inv_norm * inv_norm * inv_norm

    dot_d_ddx = dx * ddx_x + dy * ddx_y + dz * ddx_z
    dot_d_ddy = dx * ddy_x + dy * ddy_y + dz * ddy_z

    dhat_x = dx * inv_norm
    dhat_y = dy * inv_norm
    dhat_z = dz * inv_norm

    ddhatx_x = ddx_x * inv_norm - dx * dot_d_ddx * inv_norm3
    ddhatx_y = ddx_y * inv_norm - dy * dot_d_ddx * inv_norm3
    ddhatx_z = ddx_z * inv_norm - dz * dot_d_ddx * inv_norm3

    ddhaty_x = ddy_x * inv_norm - dx * dot_d_ddy * inv_norm3
    ddhaty_y = ddy_y * inv_norm - dy * dot_d_ddy * inv_norm3
    ddhaty_z = ddy_z * inv_norm - dz * dot_d_ddy * inv_norm3

    cross_x = ddhatx_y * ddhaty_z - ddhatx_z * ddhaty_y
    cross_y = ddhatx_z * ddhaty_x - ddhatx_x * ddhaty_z
    cross_z = ddhatx_x * ddhaty_y - ddhatx_y * ddhaty_x

    berry = -0.5 * (dhat_x * cross_x + dhat_y * cross_y + dhat_z * cross_z)
    metric_trace = 0.25 * (
        ddhatx_x * ddhatx_x
        + ddhatx_y * ddhatx_y
        + ddhatx_z * ddhatx_z
        + ddhaty_x * ddhaty_x
        + ddhaty_y * ddhaty_y
        + ddhaty_z * ddhaty_z
    )

    chern_number = prefactor * float(np.sum(berry))
    quantum_weight = prefactor * float(np.sum(metric_trace))
    trace_condition = quantum_weight - chern_number
    return band_width, band_gap, width_gap_ratio, chern_number, quantum_weight, trace_condition


def argmin_on_mask(values: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    # Written with Codex 02-19-26.
    if not np.any(mask):
        return None
    flat_indices = np.flatnonzero(mask.ravel())
    sub_values = values.ravel()[flat_indices]
    best_flat = int(flat_indices[int(np.argmin(sub_values))])
    return np.unravel_index(best_flat, values.shape)


def robust_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    # Written with Codex 02-19-26.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not (vmax > vmin):
        vmax = vmin + 1.0
    return vmin, vmax


def index_to_point(
    idx: tuple[int, int] | None,
    t2_values: np.ndarray,
    phi_values: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, float] | None:
    # Written with Codex 02-19-26.
    if idx is None:
        return None
    i, j = idx
    return {
        "t2": float(t2_values[i]),
        "phi": float(phi_values[j]),
        "band_width": float(metrics["band_width"][i, j]),
        "band_gap": float(metrics["band_gap"][i, j]),
        "width_gap_ratio": float(metrics["width_gap_ratio"][i, j]),
        "chern_number": float(metrics["chern_number"][i, j]),
        "quantum_weight": float(metrics["quantum_weight"][i, j]),
        "trace_condition": float(metrics["trace_condition"][i, j]),
    }


def save_csv(
    path: Path,
    t2_values: np.ndarray,
    phi_values: np.ndarray,
    metrics: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    topological_mask: np.ndarray,
) -> None:
    # Written with Codex 02-19-26.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t2",
                "phi",
                "band_width",
                "band_gap",
                "width_gap_ratio",
                "chern_number",
                "quantum_weight",
                "trace_condition",
                "valid_gapped",
                "chern_near_target",
            ]
        )
        for i, t2 in enumerate(t2_values):
            for j, phi in enumerate(phi_values):
                writer.writerow(
                    [
                        float(t2),
                        float(phi),
                        float(metrics["band_width"][i, j]),
                        float(metrics["band_gap"][i, j]),
                        float(metrics["width_gap_ratio"][i, j]),
                        float(metrics["chern_number"][i, j]),
                        float(metrics["quantum_weight"][i, j]),
                        float(metrics["trace_condition"][i, j]),
                        int(valid_mask[i, j]),
                        int(topological_mask[i, j]),
                    ]
                )


def plot_heatmaps(
    out_path: Path,
    t2_values: np.ndarray,
    phi_values: np.ndarray,
    ratio_grid: np.ndarray,
    trace_grid: np.ndarray,
    best_ratio_idx: tuple[int, int] | None,
    best_trace_idx: tuple[int, int] | None,
    best_balanced_idx: tuple[int, int] | None,
) -> None:
    # Written with Codex 02-19-26.
    phi_min = float(phi_values[0])
    phi_max = float(phi_values[-1])
    t2_min = float(t2_values[0])
    t2_max = float(t2_values[-1])
    extent = (phi_min, phi_max, t2_min, t2_max)

    ratio_plot = np.where(np.isfinite(ratio_grid), ratio_grid, np.nan)
    trace_plot = np.where(np.isfinite(trace_grid), trace_grid, np.nan)

    ratio_vmin, ratio_vmax = robust_limits(ratio_plot)
    trace_vmin, trace_vmax = robust_limits(trace_plot)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    cmap_ratio = plt.get_cmap("viridis").copy()
    cmap_trace = plt.get_cmap("magma").copy()
    cmap_ratio.set_bad(color="lightgray")
    cmap_trace.set_bad(color="lightgray")

    im0 = axes[0].imshow(
        np.ma.masked_invalid(ratio_plot),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_ratio,
        vmin=ratio_vmin,
        vmax=ratio_vmax,
    )
    axes[0].set_title("Lowest-band width / bandgap")
    axes[0].set_xlabel(r"$\phi$ (rad)")
    axes[0].set_ylabel(r"$t_2$")
    fig.colorbar(im0, ax=axes[0], label="width/gap")

    im1 = axes[1].imshow(
        np.ma.masked_invalid(trace_plot),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_trace,
        vmin=trace_vmin,
        vmax=trace_vmax,
    )
    axes[1].set_title(r"Trace condition: $Q - C$")
    axes[1].set_xlabel(r"$\phi$ (rad)")
    axes[1].set_ylabel(r"$t_2$")
    fig.colorbar(im1, ax=axes[1], label=r"$Q - C$")

    if best_ratio_idx is not None:
        i, j = best_ratio_idx
        axes[0].scatter(phi_values[j], t2_values[i], s=70, c="white", edgecolors="black", marker="o")
    if best_trace_idx is not None:
        i, j = best_trace_idx
        axes[1].scatter(phi_values[j], t2_values[i], s=70, c="white", edgecolors="black", marker="o")
    if best_balanced_idx is not None:
        i, j = best_balanced_idx
        for ax in axes:
            ax.scatter(phi_values[j], t2_values[i], s=90, c="none", edgecolors="cyan", marker="s", linewidths=1.7)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # Written with Codex 02-19-26.
    args = parse_args()

    job_dir = args.job_dir.resolve()
    scripts_dir = job_dir / "scripts"
    raw_data_dir = job_dir / "raw_data"
    plots_dir = job_dir / "plots"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    a1 = np.asarray(PRIMITIVE_A1, dtype=np.float64)
    a2 = np.asarray(PRIMITIVE_A2, dtype=np.float64)
    b1, b2 = reciprocal_vectors(a1, a2)

    k_bz_all = sample_shortest_representative_bz(
        args.lx_bz_plot,
        args.lx_bz_plot,
        b1,
        b2,
        unique=False,
    )
    if k_bz_all.size == 0:
        raise RuntimeError("No BZ points sampled. Increase lx-bz-plot.")

    tables = precompute_k_tables(k_bz_all)
    n_k = int(k_bz_all.shape[0])
    area_bz = float(abs(b1[0] * b2[1] - b1[1] * b2[0]))
    prefactor = area_bz / (n_k * 2.0 * np.pi)

    t2_values = np.linspace(args.t2_min, args.t2_max, args.n_t2, dtype=np.float64)
    phi_values = np.linspace(args.phi_min, args.phi_max, args.n_phi, dtype=np.float64)

    shape = (t2_values.size, phi_values.size)
    band_width = np.full(shape, np.nan, dtype=np.float64)
    band_gap = np.full(shape, np.nan, dtype=np.float64)
    width_gap_ratio = np.full(shape, np.inf, dtype=np.float64)
    chern_number = np.full(shape, np.nan, dtype=np.float64)
    quantum_weight = np.full(shape, np.nan, dtype=np.float64)
    trace_condition = np.full(shape, np.nan, dtype=np.float64)

    for i, t2 in enumerate(t2_values):
        for j, phi in enumerate(phi_values):
            (
                bw,
                bg,
                ratio,
                chern,
                qweight,
                trace_cond,
            ) = evaluate_metrics_at_parameters(
                t2=float(t2),
                phi=float(phi),
                m=float(args.m),
                tables=tables,
                prefactor=prefactor,
                gap_tol=float(args.gap_tol),
            )
            band_width[i, j] = bw
            band_gap[i, j] = bg
            width_gap_ratio[i, j] = ratio
            chern_number[i, j] = chern
            quantum_weight[i, j] = qweight
            trace_condition[i, j] = trace_cond

    metrics = {
        "band_width": band_width,
        "band_gap": band_gap,
        "width_gap_ratio": width_gap_ratio,
        "chern_number": chern_number,
        "quantum_weight": quantum_weight,
        "trace_condition": trace_condition,
    }

    valid_mask = (
        np.isfinite(width_gap_ratio)
        & np.isfinite(trace_condition)
        & np.isfinite(chern_number)
        & np.isfinite(quantum_weight)
        & (band_gap > args.gap_tol)
    )
    topological_mask = valid_mask & (np.abs(chern_number - args.chern_target) <= args.chern_tol)

    ratio_min_idx = argmin_on_mask(width_gap_ratio, valid_mask)
    trace_min_idx = argmin_on_mask(trace_condition, valid_mask)

    balanced_idx = None
    if np.any(valid_mask):
        ratio_vals = width_gap_ratio[valid_mask]
        trace_vals = trace_condition[valid_mask]
        ratio_span = float(np.max(ratio_vals) - np.min(ratio_vals))
        trace_span = float(np.max(trace_vals) - np.min(trace_vals))
        ratio_norm = (width_gap_ratio - np.min(ratio_vals)) / (ratio_span if ratio_span > 0.0 else 1.0)
        trace_norm = (trace_condition - np.min(trace_vals)) / (trace_span if trace_span > 0.0 else 1.0)
        balanced_score = ratio_norm + trace_norm
        balanced_idx = argmin_on_mask(balanced_score, valid_mask)

    balanced_topo_idx = None
    if np.any(topological_mask):
        ratio_vals = width_gap_ratio[topological_mask]
        trace_vals = trace_condition[topological_mask]
        ratio_span = float(np.max(ratio_vals) - np.min(ratio_vals))
        trace_span = float(np.max(trace_vals) - np.min(trace_vals))
        ratio_norm = (width_gap_ratio - np.min(ratio_vals)) / (ratio_span if ratio_span > 0.0 else 1.0)
        trace_norm = (trace_condition - np.min(trace_vals)) / (trace_span if trace_span > 0.0 else 1.0)
        balanced_score_topo = ratio_norm + trace_norm
        balanced_topo_idx = argmin_on_mask(balanced_score_topo, topological_mask)

    np.savez_compressed(
        raw_data_dir / "sweep_metrics.npz",
        t2_values=t2_values,
        phi_values=phi_values,
        band_width=band_width,
        band_gap=band_gap,
        width_gap_ratio=width_gap_ratio,
        chern_number=chern_number,
        quantum_weight=quantum_weight,
        trace_condition=trace_condition,
        valid_mask=valid_mask,
        topological_mask=topological_mask,
        config=np.array(
            [
                args.t1,
                args.m,
                args.lx_bz_plot,
                args.gap_tol,
                args.chern_target,
                args.chern_tol,
            ],
            dtype=np.float64,
        ),
    )

    save_csv(
        raw_data_dir / "sweep_metrics.csv",
        t2_values=t2_values,
        phi_values=phi_values,
        metrics=metrics,
        valid_mask=valid_mask,
        topological_mask=topological_mask,
    )

    plot_heatmaps(
        plots_dir / "t2_phi_objective_heatmaps.png",
        t2_values=t2_values,
        phi_values=phi_values,
        ratio_grid=width_gap_ratio,
        trace_grid=trace_condition,
        best_ratio_idx=ratio_min_idx,
        best_trace_idx=trace_min_idx,
        best_balanced_idx=balanced_topo_idx if balanced_topo_idx is not None else balanced_idx,
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "job_dir": str(job_dir),
        "config": {
            "t1": float(args.t1),
            "m": float(args.m),
            "t2_min": float(args.t2_min),
            "t2_max": float(args.t2_max),
            "n_t2": int(args.n_t2),
            "phi_min": float(args.phi_min),
            "phi_max": float(args.phi_max),
            "n_phi": int(args.n_phi),
            "lx_bz_plot": int(args.lx_bz_plot),
            "gap_tol": float(args.gap_tol),
            "chern_target": float(args.chern_target),
            "chern_tol": float(args.chern_tol),
        },
        "optima": {
            "min_width_gap_ratio": index_to_point(ratio_min_idx, t2_values, phi_values, metrics),
            "min_trace_condition": index_to_point(trace_min_idx, t2_values, phi_values, metrics),
            "balanced_global": index_to_point(balanced_idx, t2_values, phi_values, metrics),
            "balanced_topological": index_to_point(
                balanced_topo_idx, t2_values, phi_values, metrics
            ),
        },
        "counts": {
            "total_points": int(t2_values.size * phi_values.size),
            "valid_gapped_points": int(np.sum(valid_mask)),
            "topological_points": int(np.sum(topological_mask)),
        },
        "artifacts": {
            "raw_npz": "raw_data/sweep_metrics.npz",
            "raw_csv": "raw_data/sweep_metrics.csv",
            "heatmap_png": "plots/t2_phi_objective_heatmaps.png",
            "script": "scripts/run_t2_phi_sweep.py",
        },
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_text = "\n".join(
        [
            "Single-particle Haldane sweep in (t2, phi) at fixed t1 and m.",
            "",
            "This job stores:",
            "- raw_data/sweep_metrics.npz : full grid arrays.",
            "- raw_data/sweep_metrics.csv : long-form table (one row per parameter point).",
            "- plots/t2_phi_objective_heatmaps.png : parameter-space heatmaps.",
            "- summary.json : run configuration and reported optima.",
            "- scripts/run_t2_phi_sweep.py : exact script used for this run.",
        ]
    )
    (job_dir / "README.txt").write_text(readme_text + "\n", encoding="utf-8")

    best_global = summary["optima"]["balanced_global"]
    best_topo = summary["optima"]["balanced_topological"]
    best_print = best_topo if best_topo is not None else best_global

    print(f"Saved results to: {job_dir}")
    print(f"Total points: {summary['counts']['total_points']}")
    print(f"Valid gapped points: {summary['counts']['valid_gapped_points']}")
    print(f"Topological points (|C-{args.chern_target}| <= {args.chern_tol}): {summary['counts']['topological_points']}")
    if best_print is not None:
        print(
            "Recommended optimum "
            f"(t2={best_print['t2']:.6f}, phi={best_print['phi']:.6f}): "
            f"width/gap={best_print['width_gap_ratio']:.6f}, "
            f"trace_condition={best_print['trace_condition']:.6f}, "
            f"C={best_print['chern_number']:.6f}, Q={best_print['quantum_weight']:.6f}"
        )
    else:
        print("No valid optimum found in this sweep window.")


if __name__ == "__main__":
    main()
