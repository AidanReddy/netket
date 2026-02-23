from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aidan_custom.geometry import (
    first_bz_hexagon_vertices,
    high_symmetry_points,
    k_path_gamma_k_m_kp_gamma,
    sample_shortest_representative_bz,
)
from aidan_custom.mote2_three_orbital import (
    mote2_three_orbital_berry_and_metric_trace,
    mote2_three_orbital_eigenvalues,
    mote2_three_orbital_reciprocal_vectors,
)

PARAM_COLUMNS = ["delta", "t_hh1", "t_th2", "t_hh3", "t_tt1"]
ANCHOR = {
    "delta": 27.9 / 5.15,
    "t_hh1": 1.81 / 5.15,
    "t_th2": 0.07 / 5.15,
    "t_hh3": 0.43 / 5.15,
    "t_tt1": -0.46 / 5.15,
}
LOCAL_BOUNDS: dict[str, tuple[float, float]] = {
    "delta": (4.4, 6.4),
    "t_hh1": (0.20, 0.55),
    "t_th2": (-0.07, 0.09),
    "t_hh3": (0.00, 0.17),
    "t_tt1": (-0.18, 0.02),
}

GAP_FLOOR = 0.05
CHERN_TOL = 0.05


def _clip_to_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    out = df.copy()
    for name, (lo, hi) in bounds.items():
        out[name] = np.clip(out[name].to_numpy(float), lo, hi)
    return out


def _deduplicate_parameters(df: pd.DataFrame, decimals: int = 9) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    rounded = df[PARAM_COLUMNS].round(decimals)
    keep = ~rounded.duplicated()
    return df.loc[keep].reset_index(drop=True)


def _sample_stage1(
    rng: np.random.Generator,
    n_samples: int,
    anchor: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    rows: list[dict[str, float]] = []
    for _ in range(int(max(0, n_samples))):
        row: dict[str, float] = {}
        for name, (lo, hi) in bounds.items():
            if rng.random() < 0.80:
                sigma = (hi - lo) / 10.0
                value = float(anchor[name] + rng.normal(0.0, sigma))
            else:
                value = float(rng.uniform(lo, hi))
            row[name] = float(np.clip(value, lo, hi))
        rows.append(row)

    return pd.DataFrame(rows, columns=PARAM_COLUMNS)


def _sample_local(
    rng: np.random.Generator,
    seeds: pd.DataFrame,
    n_per_seed: int,
    scales: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    if len(seeds) == 0 or int(n_per_seed) <= 0:
        return pd.DataFrame(columns=PARAM_COLUMNS)

    seed_vals = seeds[PARAM_COLUMNS].to_numpy(float)
    rows: list[dict[str, float]] = []
    for seed in seed_vals:
        for _ in range(int(n_per_seed)):
            row: dict[str, float] = {}
            for j, name in enumerate(PARAM_COLUMNS):
                lo, hi = bounds[name]
                value = float(seed[j] + rng.normal(0.0, scales[name]))
                row[name] = float(np.clip(value, lo, hi))
            rows.append(row)

    return pd.DataFrame(rows, columns=PARAM_COLUMNS)


def _k_meshes(nk: int, a_m: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=a_m)
    k_regular = sample_shortest_representative_bz(nk, nk, b1, b2, unique=False)

    coeff = (np.arange(nk, dtype=float) + 0.5) / float(nk)
    c1, c2 = np.meshgrid(coeff, coeff, indexing="ij")
    k_shift = np.empty((nk * nk, 2), dtype=float)
    k_shift[:, 0] = c1.ravel() * b1[0] + c2.ravel() * b2[0]
    k_shift[:, 1] = c1.ravel() * b1[1] + c2.ravel() * b2[1]

    area_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    prefactor = area_bz / (2.0 * np.pi * nk * nk)
    return k_regular, k_shift, prefactor, b1, b2


def _evaluate_samples(param_df: pd.DataFrame, nk: int, gap_floor: float) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    k_regular, k_shift, prefactor, _, _ = _k_meshes(nk=nk, a_m=1.0)
    k_gap = np.concatenate([k_regular, k_shift], axis=0)

    n = len(param_df)
    bandwidth = np.full(n, np.nan, dtype=float)
    gap_min = np.full(n, np.nan, dtype=float)
    flatness = np.full(n, np.nan, dtype=float)
    chern = np.full(n, np.nan, dtype=float)
    qweight = np.full(n, np.nan, dtype=float)

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
        g_min = np.inf
        for kx, ky in k_gap:
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
                ph_conj=True,
            )
            e_low = float(evals[0])
            e_mid = float(evals[1])
            if e_low < e_min:
                e_min = e_low
            if e_low > e_max:
                e_max = e_low
            gap = e_mid - e_low
            if gap < g_min:
                g_min = gap

        bw = e_max - e_min
        bandwidth[i] = bw
        gap_min[i] = g_min

        if g_min <= float(gap_floor):
            flatness[i] = np.inf
            continue

        flatness[i] = bw / g_min

        berry_sum = 0.0
        metric_sum = 0.0
        failed = False
        for kx, ky in k_regular:
            try:
                berry, metric = geo_fn(
                    kx,
                    ky,
                    "lowest",
                    1.0e-5,
                    1.0e-8,
                    delta,
                    0.0,
                    1.0,
                    t_hh1,
                    t_th2,
                    t_hh3,
                    t_tt1,
                    1.0,
                    ph_conj=True,
                )
            except ValueError:
                failed = True
                break

            berry_sum += berry
            metric_sum += metric

        if failed:
            flatness[i] = np.inf
            continue

        chern[i] = prefactor * berry_sum
        qweight[i] = prefactor * metric_sum

    out = param_df.copy()
    out["nk"] = int(nk)
    out["bandwidth_low"] = bandwidth
    out["gap_low_mid"] = gap_min
    out["flatness_ratio"] = flatness
    out["chern_low"] = chern
    out["quantum_weight_low"] = qweight
    out["trace_violation"] = qweight - chern
    out["trace_violation_abs"] = np.abs(out["trace_violation"])
    out["chern_error"] = np.abs(chern + 1.0)
    out["chern_mag_error"] = np.abs(np.abs(chern) - 1.0)
    out["is_gapped"] = gap_min > float(gap_floor)
    return out


def _candidate_mask(df: pd.DataFrame, chern_target: float, chern_tol: float, gap_floor: float) -> np.ndarray:
    # Written with Codex 02-20-26.
    return (
        df["is_gapped"].to_numpy(bool)
        & np.isfinite(df["flatness_ratio"].to_numpy(float))
        & np.isfinite(df["trace_violation"].to_numpy(float))
        & (np.abs(df["chern_low"].to_numpy(float) - chern_target) <= float(chern_tol))
        & (df["gap_low_mid"].to_numpy(float) > float(gap_floor))
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    # Written with Codex 02-20-26.
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.inf, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    span = hi - lo
    if span <= 0.0:
        out[finite] = 0.0
        return out
    out[finite] = (arr[finite] - lo) / span
    return out


def _score(df: pd.DataFrame, chern_target: float) -> np.ndarray:
    # Written with Codex 02-20-26.
    flat = _normalized(df["flatness_ratio"].to_numpy(float))
    trace = _normalized(df["trace_violation"].to_numpy(float))
    chern_err = _normalized(np.abs(df["chern_low"].to_numpy(float) - chern_target))
    return flat + trace + 1.5 * chern_err


def _select_seeds(
    df: pd.DataFrame,
    n_seeds: int,
    chern_target: float,
    chern_tol: float,
    gap_floor: float,
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    mask = _candidate_mask(df, chern_target=chern_target, chern_tol=chern_tol, gap_floor=gap_floor)
    candidates = df.loc[mask].copy()
    if len(candidates) == 0:
        raise RuntimeError("No candidates survived gap/Chern filtering in coarse stage.")

    candidates["score"] = _score(candidates, chern_target=chern_target)
    seeds = candidates.sort_values(["score", "flatness_ratio", "trace_violation"]).head(int(n_seeds))
    return seeds[PARAM_COLUMNS].reset_index(drop=True)


def _plot_landscape(
    df_all: pd.DataFrame,
    df_good: pd.DataFrame,
    best_flat: pd.DataFrame,
    best_trace: pd.DataFrame,
    out_dir: Path,
    chern_target: float,
) -> None:
    # Written with Codex 02-20-26.
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    mask = np.isfinite(df_all["flatness_ratio"]) & np.isfinite(df_all["trace_violation"])
    df_plot = df_all.loc[mask]

    sc = ax.scatter(
        df_plot["flatness_ratio"],
        df_plot["trace_violation"],
        c=df_plot["chern_low"],
        s=24,
        alpha=0.75,
        cmap="coolwarm",
    )

    ax.scatter(
        df_good["flatness_ratio"],
        df_good["trace_violation"],
        facecolors="none",
        edgecolors="black",
        s=58,
        linewidths=0.8,
        label=f"|C-{chern_target:.0f}| <= {CHERN_TOL:.2f}",
    )

    r_flat = best_flat.iloc[0]
    r_trace = best_trace.iloc[0]
    ax.scatter(
        [r_flat["flatness_ratio"]],
        [r_flat["trace_violation"]],
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        s=220,
        label="Best flatness",
    )
    ax.scatter(
        [r_trace["flatness_ratio"]],
        [r_trace["trace_violation"]],
        marker="D",
        color="limegreen",
        edgecolors="black",
        linewidths=0.8,
        s=80,
        label="Best trace",
    )

    ax.set_xlabel("Bandwidth / gap (lowest to second-lowest)")
    ax.set_ylabel("Trace condition violation: Q - C")
    ax.set_title("Anchor-local MoTe2 search (lowest band, ph_conj=True)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="Chern (lowest)")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_scatter_anchor_local.png", dpi=180)
    plt.close(fig)


def _candidate_params(row: pd.Series) -> dict[str, float]:
    # Written with Codex 02-20-26.
    return {
        "delta": float(row["delta"]),
        "ez": 0.0,
        "t_th1": 1.0,
        "t_hh1": float(row["t_hh1"]),
        "t_th2": float(row["t_th2"]),
        "t_hh3": float(row["t_hh3"]),
        "t_tt1": float(row["t_tt1"]),
        "a_m": 1.0,
    }


def _plot_candidate_band_and_geometry(
    row: pd.Series,
    out_path: Path,
    n_path: int = 180,
    nk_geo: int = 41,
) -> None:
    # Written with Codex 02-20-26.
    params = _candidate_params(row)

    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=params["a_m"])
    k_path, x, x_nodes, labels = k_path_gamma_k_m_kp_gamma(n_path, b1, b2, fold=True)
    bands = np.array(
        [mote2_three_orbital_eigenvalues(kx, ky, ph_conj=True, **params) for kx, ky in k_path],
        dtype=float,
    )

    gamma, k_node, m_node, kp_node = high_symmetry_points(b1, b2)
    bz_hex = first_bz_hexagon_vertices(b1, b2)
    k_points = sample_shortest_representative_bz(nk_geo, nk_geo, b1, b2, unique=True)

    berry = np.empty(k_points.shape[0], dtype=float)
    metric_trace = np.empty(k_points.shape[0], dtype=float)
    for i, (kx, ky) in enumerate(k_points):
        bcurv, mtr = mote2_three_orbital_berry_and_metric_trace(
            kx=kx,
            ky=ky,
            band="lowest",
            ph_conj=True,
            **params,
        )
        berry[i] = bcurv
        metric_trace[i] = mtr

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8), constrained_layout=True)

    ax = axes[0]
    for n in range(3):
        ax.plot(x, bands[:, n], lw=1.6, color="blue")
    for xn in x_nodes:
        ax.axvline(xn, color="0.7", lw=0.8)
    ax.set_xticks(x_nodes)
    ax.set_xticklabels(labels)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylabel("Energy [arb.]")
    ax.set_title("Band structure")
    ax.grid(axis="y", alpha=0.2)

    hex_x = np.r_[bz_hex[:, 0], bz_hex[0, 0]]
    hex_y = np.r_[bz_hex[:, 1], bz_hex[0, 1]]
    special = np.array([gamma, k_node, m_node, kp_node])

    ax = axes[1]
    bmax = max(float(np.max(np.abs(berry))), 1.0e-12)
    sc0 = ax.scatter(
        k_points[:, 0],
        k_points[:, 1],
        c=berry,
        s=20,
        cmap="jet_r",
        vmin=-bmax,
        vmax=bmax,
    )
    ax.plot(hex_x, hex_y, color="black", lw=1.0, zorder=-1)
    ax.scatter(special[:, 0], special[:, 1], color="black", s=16)
    ax.set_aspect("equal")
    ax.set_title("Berry curvature")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    fig.colorbar(sc0, ax=ax, shrink=0.84)

    ax = axes[2]
    gmax = max(float(np.max(metric_trace)), 1.0e-12)
    sc1 = ax.scatter(
        k_points[:, 0],
        k_points[:, 1],
        c=metric_trace,
        s=20,
        cmap="jet",
        vmin=0.0,
        vmax=gmax,
    )
    ax.plot(hex_x, hex_y, color="black", lw=1.0, zorder=-1)
    ax.scatter(special[:, 0], special[:, 1], color="black", s=16)
    ax.set_aspect("equal")
    ax.set_title("Metric trace tr(g)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    fig.colorbar(sc1, ax=ax, shrink=0.84)

    fig.suptitle(
        (
            f"W/Delta={float(row['flatness_ratio']):.6f}, gap={float(row['gap_low_mid']):.6f}, "
            f"C={float(row['chern_low']):.6f}, Q-C={float(row['trace_violation']):.6f}; "
            f"(delta,t_hh1,t_th2,t_hh3,t_tt1)=({float(row['delta']):.4f},{float(row['t_hh1']):.4f},"
            f"{float(row['t_th2']):.4f},{float(row['t_hh3']):.4f},{float(row['t_tt1']):.4f})"
        ),
        fontsize=10,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    # Written with Codex 02-20-26.
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(20260220)

    nk_eval = 21
    n_stage1 = 1200
    n_seed = 140
    n_local_per_seed = 3

    print("Evaluating theta=3deg anchor baseline ...", flush=True)
    anchor_df = pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS)
    anchor_eval = _evaluate_samples(anchor_df, nk=nk_eval, gap_floor=GAP_FLOOR).iloc[0]
    chern_target = float(np.rint(anchor_eval["chern_low"]))

    print(
        f"Anchor: W/Delta={float(anchor_eval['flatness_ratio']):.6f}, "
        f"gap={float(anchor_eval['gap_low_mid']):.6f}, C={float(anchor_eval['chern_low']):.6f}, "
        f"Q-C={float(anchor_eval['trace_violation']):.6f}",
        flush=True,
    )

    print("Stage 1: anchor-local sampling ...", flush=True)
    stage1_params = pd.concat(
        [
            pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS),
            _sample_stage1(rng=rng, n_samples=n_stage1, anchor=ANCHOR, bounds=LOCAL_BOUNDS),
        ],
        ignore_index=True,
    )
    stage1_params = _deduplicate_parameters(_clip_to_bounds(stage1_params, LOCAL_BOUNDS))
    stage1_eval = _evaluate_samples(stage1_params, nk=nk_eval, gap_floor=GAP_FLOOR)

    print("Stage 2: local refinement around best seeds ...", flush=True)
    seed_df = _select_seeds(
        df=stage1_eval,
        n_seeds=n_seed,
        chern_target=chern_target,
        chern_tol=CHERN_TOL,
        gap_floor=GAP_FLOOR,
    )
    local_scales = {
        "delta": 0.10,
        "t_hh1": 0.020,
        "t_th2": 0.012,
        "t_hh3": 0.012,
        "t_tt1": 0.015,
    }
    stage2_params = pd.concat(
        [
            pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS),
            seed_df,
            _sample_local(
                rng=rng,
                seeds=seed_df,
                n_per_seed=n_local_per_seed,
                scales=local_scales,
                bounds=LOCAL_BOUNDS,
            ),
        ],
        ignore_index=True,
    )
    stage2_params = _deduplicate_parameters(_clip_to_bounds(stage2_params, LOCAL_BOUNDS))
    stage2_eval = _evaluate_samples(stage2_params, nk=nk_eval, gap_floor=GAP_FLOOR)

    all_eval = pd.concat([stage1_eval, stage2_eval], ignore_index=True)
    all_eval = _deduplicate_parameters(all_eval)

    good_mask = _candidate_mask(
        all_eval,
        chern_target=chern_target,
        chern_tol=CHERN_TOL,
        gap_floor=GAP_FLOOR,
    )
    good_eval = all_eval.loc[good_mask].copy()
    if len(good_eval) == 0:
        raise RuntimeError("No candidates survived the strict local filter.")

    good_eval["score"] = _score(good_eval, chern_target=chern_target)

    best_flat = good_eval.sort_values(["flatness_ratio", "trace_violation", "chern_mag_error"]).head(1)
    best_trace = good_eval.sort_values(["trace_violation", "flatness_ratio", "chern_mag_error"]).head(1)

    good_sorted = good_eval.sort_values(["score", "flatness_ratio", "trace_violation"]).copy()
    top_candidates = good_sorted.head(8).copy()

    stage1_eval.to_csv(out_dir / "stage1_anchor_local.csv", index=False)
    stage2_eval.to_csv(out_dir / "stage2_anchor_refined.csv", index=False)
    all_eval.to_csv(out_dir / "all_candidates_evaluated.csv", index=False)
    good_sorted.to_csv(out_dir / "strict_candidates_sorted.csv", index=False)
    top_candidates.to_csv(out_dir / "top_candidates.csv", index=False)
    best_flat.to_csv(out_dir / "best_by_flatness.csv", index=False)
    best_trace.to_csv(out_dir / "best_by_trace_violation.csv", index=False)

    _plot_landscape(
        df_all=all_eval,
        df_good=good_eval,
        best_flat=best_flat,
        best_trace=best_trace,
        out_dir=out_dir,
        chern_target=chern_target,
    )

    selected_rows = pd.concat([best_flat, best_trace, good_sorted.head(3)], ignore_index=True)
    selected_rows = _deduplicate_parameters(selected_rows)
    selected_rows = selected_rows.head(3).reset_index(drop=True)
    for i, row in selected_rows.iterrows():
        out_png = out_dir / f"candidate_{i+1:02d}_band_and_quantum_geometry.png"
        _plot_candidate_band_and_geometry(row=row, out_path=out_png)

    summary = {
        "seed": 20260220,
        "search_type": "anchor_local_theta3deg",
        "ph_conj": True,
        "band": "lowest",
        "bandgap_definition": "E_band1 - E_band0",
        "nk_evaluation": nk_eval,
        "gap_floor": float(GAP_FLOOR),
        "chern_target": float(chern_target),
        "chern_tolerance": float(CHERN_TOL),
        "anchor": {
            key: float(anchor_eval[key])
            for key in [
                "delta",
                "t_hh1",
                "t_th2",
                "t_hh3",
                "t_tt1",
                "flatness_ratio",
                "bandwidth_low",
                "gap_low_mid",
                "chern_low",
                "quantum_weight_low",
                "trace_violation",
            ]
        },
        "counts": {
            "stage1": int(len(stage1_eval)),
            "stage2": int(len(stage2_eval)),
            "all": int(len(all_eval)),
            "strict": int(len(good_eval)),
        },
        "best_by_flatness": {
            key: float(best_flat.iloc[0][key])
            for key in [
                "delta",
                "t_hh1",
                "t_th2",
                "t_hh3",
                "t_tt1",
                "flatness_ratio",
                "bandwidth_low",
                "gap_low_mid",
                "chern_low",
                "quantum_weight_low",
                "trace_violation",
            ]
        },
        "best_by_trace_violation": {
            key: float(best_trace.iloc[0][key])
            for key in [
                "delta",
                "t_hh1",
                "t_th2",
                "t_hh3",
                "t_tt1",
                "flatness_ratio",
                "bandwidth_low",
                "gap_low_mid",
                "chern_low",
                "quantum_weight_low",
                "trace_violation",
            ]
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    shutil.copyfile(__file__, out_dir / "script_used_for_search.py")

    cols = [
        "delta",
        "t_hh1",
        "t_th2",
        "t_hh3",
        "t_tt1",
        "flatness_ratio",
        "gap_low_mid",
        "bandwidth_low",
        "chern_low",
        "quantum_weight_low",
        "trace_violation",
    ]
    print("\nBest by flatness:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(best_flat[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nBest by trace violation:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(best_trace[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nTop strict candidates:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(top_candidates[cols].head(6).to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nWrote files:", flush=True)
    for name in [
        "stage1_anchor_local.csv",
        "stage2_anchor_refined.csv",
        "all_candidates_evaluated.csv",
        "strict_candidates_sorted.csv",
        "top_candidates.csv",
        "best_by_flatness.csv",
        "best_by_trace_violation.csv",
        "objective_scatter_anchor_local.png",
        "candidate_01_band_and_quantum_geometry.png",
        "candidate_02_band_and_quantum_geometry.png",
        "candidate_03_band_and_quantum_geometry.png",
        "summary.json",
        "script_used_for_search.py",
    ]:
        print(f"- {out_dir / name}", flush=True)


if __name__ == "__main__":
    main()
