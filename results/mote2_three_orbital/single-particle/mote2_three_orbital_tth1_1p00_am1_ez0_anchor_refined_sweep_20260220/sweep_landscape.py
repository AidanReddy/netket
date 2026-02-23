from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aidan_custom.geometry import sample_shortest_representative_bz
from aidan_custom.mote2_three_orbital import (
    mote2_three_orbital_berry_and_metric_trace,
    mote2_three_orbital_eigenvalues,
    mote2_three_orbital_reciprocal_vectors,
)


def _anchor_parameters() -> dict[str, float]:
    # Written with Codex 02-20-26.
    return {
        "delta": 27.9 / 5.15,
        "t_hh1": 1.81 / 5.15,
        "t_th2": 0.07 / 5.15,
        "t_hh3": 0.43 / 5.15,
        "t_tt1": -0.46 / 5.15,
    }


def _make_parameter_samples(rng: np.random.Generator, n_random: int, n_local: int) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    ranges_global = {
        "delta": (2.0, 8.0),
        "t_hh1": (0.05, 1.20),
        "t_th2": (-0.35, 0.35),
        "t_hh3": (-0.35, 0.35),
        "t_tt1": (-0.50, 0.50),
    }
    ranges_local = {
        "delta": (4.5, 6.5),
        "t_hh1": (0.20, 0.55),
        "t_th2": (-0.12, 0.12),
        "t_hh3": (-0.05, 0.18),
        "t_tt1": (-0.25, 0.10),
    }

    n_local = int(max(0, min(n_random, n_local)))
    n_global = int(max(0, n_random - n_local))

    global_data = {name: rng.uniform(low, high, size=n_global) for name, (low, high) in ranges_global.items()}
    local_data = {name: rng.uniform(low, high, size=n_local) for name, (low, high) in ranges_local.items()}

    anchor_row = pd.DataFrame([_anchor_parameters()])
    df_global = pd.DataFrame(global_data)
    df_local = pd.DataFrame(local_data)
    return pd.concat([anchor_row, df_local, df_global], ignore_index=True)


def _k_mesh(nk: int, a_m: float = 1.0) -> tuple[np.ndarray, float]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=a_m)
    k_points = sample_shortest_representative_bz(nk, nk, b1, b2, unique=False)
    area_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    prefactor = area_bz / (2.0 * np.pi * nk * nk)
    return k_points, prefactor


def _evaluate_samples(param_df: pd.DataFrame, nk: int) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    k_points, prefactor = _k_mesh(nk=nk, a_m=1.0)

    n = len(param_df)
    bandwidth = np.empty(n, dtype=float)
    min_gap = np.empty(n, dtype=float)
    flatness = np.empty(n, dtype=float)
    chern = np.empty(n, dtype=float)
    qweight = np.empty(n, dtype=float)

    delta_arr = param_df["delta"].to_numpy(float)
    t_hh1_arr = param_df["t_hh1"].to_numpy(float)
    t_th2_arr = param_df["t_th2"].to_numpy(float)
    t_hh3_arr = param_df["t_hh3"].to_numpy(float)
    t_tt1_arr = param_df["t_tt1"].to_numpy(float)

    eig_fn = mote2_three_orbital_eigenvalues
    geo_fn = mote2_three_orbital_berry_and_metric_trace

    for i in range(n):
        delta = float(delta_arr[i])
        t_hh1 = float(t_hh1_arr[i])
        t_th2 = float(t_th2_arr[i])
        t_hh3 = float(t_hh3_arr[i])
        t_tt1 = float(t_tt1_arr[i])

        e_min = np.inf
        e_max = -np.inf
        gap_min = np.inf
        berry_sum = 0.0
        metric_sum = 0.0

        for kx, ky in k_points:
            evals = eig_fn(
                kx,
                ky,
                delta,
                0.0,
                1.0,
                t_hh1,
                t_th2,
                t_hh3,
                t_tt1,
                1.0,
            )
            e_hi = float(evals[2])
            e_mid = float(evals[1])
            if e_hi < e_min:
                e_min = e_hi
            if e_hi > e_max:
                e_max = e_hi

            gap = e_hi - e_mid
            if gap < gap_min:
                gap_min = gap

            berry, metric = geo_fn(
                kx,
                ky,
                "highest",
                1.0e-5,
                1.0e-10,
                delta,
                0.0,
                1.0,
                t_hh1,
                t_th2,
                t_hh3,
                t_tt1,
                1.0,
            )
            berry_sum += berry
            metric_sum += metric

        bw = e_max - e_min
        bandwidth[i] = bw
        min_gap[i] = gap_min
        flatness[i] = np.inf if gap_min <= 1.0e-12 else bw / gap_min
        chern[i] = prefactor * berry_sum
        qweight[i] = prefactor * metric_sum

    out = param_df.copy()
    out["nk"] = int(nk)
    out["bandwidth_high"] = bandwidth
    out["gap_high_mid"] = min_gap
    out["flatness_ratio"] = flatness
    out["chern_high"] = chern
    out["quantum_weight_high"] = qweight
    out["qw_minus_one"] = qweight - 1.0
    out["qw_gap_abs"] = np.abs(out["qw_minus_one"])
    out["chern_error"] = np.abs(chern - 1.0)
    out["is_gapped"] = min_gap > 1.0e-8
    return out


def _combined_score(df: pd.DataFrame) -> np.ndarray:
    # Written with Codex 02-20-26.
    flat = np.asarray(df["flatness_ratio"], dtype=float)
    qdiff_abs = np.asarray(df["qw_gap_abs"], dtype=float)
    cerr = np.asarray(df["chern_error"], dtype=float)

    flat_ref = max(np.percentile(flat[np.isfinite(flat)], 25), 1.0e-8)
    q_ref = max(np.percentile(qdiff_abs[np.isfinite(qdiff_abs)], 25), 1.0e-8)

    return flat / flat_ref + qdiff_abs / q_ref + 5.0 * cerr


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    # Written with Codex 02-20-26.
    n = values.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False
            continue
        dominated_by_i = np.all(values[i] <= values, axis=1) & np.any(values[i] < values, axis=1)
        dominated_by_i[i] = False
        mask[dominated_by_i] = False
    return mask


def _make_plots(df_refine: pd.DataFrame, df_top: pd.DataFrame, out_dir: Path) -> None:
    # Written with Codex 02-20-26.
    cmap = "viridis"

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(
        df_refine["flatness_ratio"],
        df_refine["qw_gap_abs"],
        c=df_refine["chern_high"],
        s=26,
        cmap=cmap,
        alpha=0.8,
    )
    ax.scatter(
        df_top["flatness_ratio"],
        df_top["qw_gap_abs"],
        facecolors="none",
        edgecolors="red",
        s=90,
        linewidths=1.5,
        label="Selected candidates",
    )
    ax.set_xlabel("Flatness ratio W/Delta (highest band)")
    ax.set_ylabel("|Quantum weight - 1|")
    ax.set_title("Objective Landscape")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="Estimated Chern (highest band)")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_landscape.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(
        df_refine["delta"],
        df_refine["t_hh1"],
        c=np.clip(df_refine["flatness_ratio"], 0.0, np.percentile(df_refine["flatness_ratio"], 95)),
        s=26,
        cmap="plasma",
        alpha=0.85,
    )
    ax.set_xlabel("delta")
    ax.set_ylabel("t_hh1")
    ax.set_title("Parameter Landscape: delta vs t_hh1")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="Flatness ratio W/Delta")
    fig.tight_layout()
    fig.savefig(out_dir / "landscape_delta_vs_t_hh1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(
        df_refine["t_th2"],
        df_refine["t_hh3"],
        c=df_refine["qw_minus_one"],
        s=26,
        cmap="cividis",
        alpha=0.85,
    )
    ax.set_xlabel("t_th2")
    ax.set_ylabel("t_hh3")
    ax.set_title("Parameter Landscape: t_th2 vs t_hh3")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="Quantum weight - 1")
    fig.tight_layout()
    fig.savefig(out_dir / "landscape_t_th2_vs_t_hh3.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(
        df_refine["delta"],
        df_refine["t_tt1"],
        c=df_refine["chern_error"],
        s=26,
        cmap="magma",
        alpha=0.85,
    )
    ax.set_xlabel("delta")
    ax.set_ylabel("t_tt1")
    ax.set_title("Parameter Landscape: delta vs t_tt1")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="|Chern - 1|")
    fig.tight_layout()
    fig.savefig(out_dir / "landscape_delta_vs_t_tt1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    # Written with Codex 02-20-26.
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(20260220)

    n_random = 1200
    n_local = 900
    nk_coarse = 21
    nk_refine = 81
    n_refine = 140

    print(f"Sampling {n_random + 1} parameter sets (including anchor; local={n_local}, global={n_random - n_local}) ...")
    params = _make_parameter_samples(rng=rng, n_random=n_random, n_local=n_local)

    print(f"Evaluating coarse metrics on {nk_coarse}x{nk_coarse} k mesh ...")
    df_coarse = _evaluate_samples(params, nk=nk_coarse)
    df_coarse["score"] = _combined_score(df_coarse)

    feasible_coarse = df_coarse[
        (df_coarse["is_gapped"]) & (df_coarse["chern_error"] <= 0.20) & np.isfinite(df_coarse["flatness_ratio"])
    ].copy()
    if len(feasible_coarse) < n_refine:
        feasible_coarse = df_coarse[df_coarse["is_gapped"] & np.isfinite(df_coarse["flatness_ratio"])].copy()

    refine_idx = feasible_coarse.nsmallest(n_refine, "score").index.tolist()
    if 0 not in refine_idx:
        if len(refine_idx) == 0:
            refine_idx = [0]
        else:
            refine_idx[-1] = 0
    refine_idx = pd.Index(refine_idx).unique()
    params_refine = params.loc[refine_idx].reset_index(drop=True)

    print(f"Refining top {len(params_refine)} candidates on {nk_refine}x{nk_refine} k mesh ...")
    df_refine = _evaluate_samples(params_refine, nk=nk_refine)
    df_refine["score"] = _combined_score(df_refine)

    # Pareto front for the two requested objectives.
    feasible_refine = df_refine[
        (df_refine["is_gapped"]) & (df_refine["chern_error"] <= 0.10) & np.isfinite(df_refine["flatness_ratio"])
    ].copy()
    if len(feasible_refine) == 0:
        feasible_refine = df_refine[df_refine["is_gapped"] & np.isfinite(df_refine["flatness_ratio"])].copy()

    obj = feasible_refine[["flatness_ratio", "qw_gap_abs"]].to_numpy(float)
    p_mask = _pareto_mask(obj)
    pareto = feasible_refine.loc[p_mask].copy()
    pareto = pareto.sort_values(["flatness_ratio", "qw_gap_abs", "chern_error"]).reset_index(drop=True)

    best = feasible_refine.sort_values(["score", "flatness_ratio", "qw_gap_abs", "chern_error"]).head(12).copy()

    df_coarse.to_csv(out_dir / "coarse_sweep.csv", index=False)
    df_refine.to_csv(out_dir / "refined_sweep.csv", index=False)
    pareto.to_csv(out_dir / "pareto_candidates.csv", index=False)
    best.to_csv(out_dir / "best_candidates.csv", index=False)

    _make_plots(df_refine=feasible_refine, df_top=best, out_dir=out_dir)

    print("\nTop candidates (refined mesh):")
    cols = [
        "delta",
        "t_hh1",
        "t_th2",
        "t_hh3",
        "t_tt1",
        "flatness_ratio",
        "gap_high_mid",
        "bandwidth_high",
        "qw_minus_one",
        "qw_gap_abs",
        "chern_high",
        "chern_error",
        "score",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nWrote files:")
    for name in [
        "coarse_sweep.csv",
        "refined_sweep.csv",
        "pareto_candidates.csv",
        "best_candidates.csv",
        "objective_landscape.png",
        "landscape_delta_vs_t_hh1.png",
        "landscape_t_th2_vs_t_hh3.png",
        "landscape_delta_vs_t_tt1.png",
    ]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
