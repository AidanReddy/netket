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
    "delta": (3.8, 7.4),
    "t_hh1": (0.12, 0.72),
    "t_th2": (-0.16, 0.18),
    "t_hh3": (-0.10, 0.26),
    "t_tt1": (-0.30, 0.12),
}

EVAL_NK = 7
GAP_FLOOR = 0.02
CHERN_MAG_TOL = 0.05
SEEDS = [20260220, 20260221, 20260222]
SCORE_WEIGHTS = {
    "flatness": 1.0,
    "trace_violation": 1.0,
    "berry_std": 1.0,
    "chern_mag_error": 0.8,
}


def _clip_to_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    out = df.copy()
    for name, (lo, hi) in bounds.items():
        out[name] = np.clip(out[name].to_numpy(float), lo, hi)
    return out


def _deduplicate_parameters(df: pd.DataFrame, decimals: int = 10) -> pd.DataFrame:
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
            if rng.random() < 0.72:
                sigma = (hi - lo) / 9.0
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
    n_k_regular = k_regular.shape[0]

    n = len(param_df)
    bandwidth = np.full(n, np.nan, dtype=float)
    gap_min = np.full(n, np.nan, dtype=float)
    flatness = np.full(n, np.nan, dtype=float)
    chern = np.full(n, np.nan, dtype=float)
    qweight = np.full(n, np.nan, dtype=float)
    berry_std = np.full(n, np.nan, dtype=float)

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
        berry_vals = np.empty(n_k_regular, dtype=float)
        failed = False
        for j, (kx, ky) in enumerate(k_regular):
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
            berry_vals[j] = berry

        if failed:
            flatness[i] = np.inf
            continue

        chern[i] = prefactor * berry_sum
        qweight[i] = prefactor * metric_sum
        berry_std[i] = float(np.std(berry_vals))

    out = param_df.copy()
    out["nk"] = int(nk)
    out["bandwidth_low"] = bandwidth
    out["gap_low_mid"] = gap_min
    out["flatness_ratio"] = flatness
    out["chern_low"] = chern
    out["quantum_weight_low"] = qweight
    out["trace_violation"] = qweight - np.abs(chern)
    out["trace_violation_abs"] = np.abs(out["trace_violation"])
    out["berry_std"] = berry_std
    out["chern_mag_error"] = np.abs(np.abs(chern) - 1.0)
    out["is_gapped"] = gap_min > float(gap_floor)
    return out


def _candidate_mask(df: pd.DataFrame, chern_mag_tol: float, gap_floor: float) -> np.ndarray:
    # Written with Codex 02-20-26.
    return (
        df["is_gapped"].to_numpy(bool)
        & np.isfinite(df["flatness_ratio"].to_numpy(float))
        & np.isfinite(df["trace_violation"].to_numpy(float))
        & np.isfinite(df["berry_std"].to_numpy(float))
        & (df["gap_low_mid"].to_numpy(float) > float(gap_floor))
        & (df["chern_mag_error"].to_numpy(float) <= float(chern_mag_tol))
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


def _score(df: pd.DataFrame) -> np.ndarray:
    # Written with Codex 02-20-26.
    flat = _normalized(df["flatness_ratio"].to_numpy(float))
    trace = _normalized(df["trace_violation"].to_numpy(float))
    berry_std = _normalized(df["berry_std"].to_numpy(float))
    chern = _normalized(df["chern_mag_error"].to_numpy(float))
    return (
        SCORE_WEIGHTS["flatness"] * flat
        + SCORE_WEIGHTS["trace_violation"] * trace
        + SCORE_WEIGHTS["berry_std"] * berry_std
        + SCORE_WEIGHTS["chern_mag_error"] * chern
    )


def _select_seed_params(df: pd.DataFrame, n_seeds: int, gap_floor: float) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    strict = df.loc[_candidate_mask(df, chern_mag_tol=CHERN_MAG_TOL, gap_floor=gap_floor)].copy()
    if len(strict) < max(12, n_seeds // 4):
        strict = df.loc[_candidate_mask(df, chern_mag_tol=0.10, gap_floor=gap_floor)].copy()
    if len(strict) == 0:
        raise RuntimeError("No seed candidates survived strict/relaxed filters.")

    strict["score"] = _score(strict)
    out = strict.sort_values(["score", "flatness_ratio", "trace_violation", "berry_std"]).head(int(n_seeds))
    return out[PARAM_COLUMNS].reset_index(drop=True)


def _run_seed_round(
    seed: int,
    anchor: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    nk_eval: int,
    gap_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Written with Codex 02-20-26.
    rng = np.random.default_rng(int(seed))

    n_stage1 = 1800
    n_seeds = 120
    n_local_per_seed = 4

    stage1_params = pd.concat(
        [
            pd.DataFrame([anchor], columns=PARAM_COLUMNS),
            _sample_stage1(rng=rng, n_samples=n_stage1, anchor=anchor, bounds=bounds),
        ],
        ignore_index=True,
    )
    stage1_params = _deduplicate_parameters(_clip_to_bounds(stage1_params, bounds))
    stage1_eval = _evaluate_samples(stage1_params, nk=nk_eval, gap_floor=gap_floor)
    stage1_eval["round_seed"] = int(seed)
    stage1_eval["stage"] = "seed_stage1"

    seeds_df = _select_seed_params(stage1_eval, n_seeds=n_seeds, gap_floor=gap_floor)
    local_scales = {
        "delta": 0.18,
        "t_hh1": 0.035,
        "t_th2": 0.020,
        "t_hh3": 0.020,
        "t_tt1": 0.025,
    }
    stage2_params = pd.concat(
        [
            pd.DataFrame([anchor], columns=PARAM_COLUMNS),
            seeds_df,
            _sample_local(
                rng=rng,
                seeds=seeds_df,
                n_per_seed=n_local_per_seed,
                scales=local_scales,
                bounds=bounds,
            ),
        ],
        ignore_index=True,
    )
    stage2_params = _deduplicate_parameters(_clip_to_bounds(stage2_params, bounds))
    stage2_eval = _evaluate_samples(stage2_params, nk=nk_eval, gap_floor=gap_floor)
    stage2_eval["round_seed"] = int(seed)
    stage2_eval["stage"] = "seed_stage2"

    return stage1_eval, stage2_eval


def _polish_round(
    all_eval: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    nk_eval: int,
    gap_floor: float,
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    strict = all_eval.loc[_candidate_mask(all_eval, chern_mag_tol=CHERN_MAG_TOL, gap_floor=gap_floor)].copy()
    strict["score"] = _score(strict)
    top = strict.sort_values(["score", "flatness_ratio", "trace_violation", "berry_std"]).head(90)

    rng = np.random.default_rng(20260230)
    local_scales = {
        "delta": 0.08,
        "t_hh1": 0.015,
        "t_th2": 0.010,
        "t_hh3": 0.010,
        "t_tt1": 0.012,
    }
    params = pd.concat(
        [
            top[PARAM_COLUMNS],
            _sample_local(
                rng=rng,
                seeds=top[PARAM_COLUMNS],
                n_per_seed=4,
                scales=local_scales,
                bounds=bounds,
            ),
            _sample_stage1(rng=rng, n_samples=140, anchor=ANCHOR, bounds=bounds),
        ],
        ignore_index=True,
    )
    params = _deduplicate_parameters(_clip_to_bounds(params, bounds))
    out = _evaluate_samples(params, nk=nk_eval, gap_floor=gap_floor)
    out["round_seed"] = 20260230
    out["stage"] = "polish"
    return out


def _plot_objective(df_all: pd.DataFrame, df_good: pd.DataFrame, out_dir: Path) -> None:
    # Written with Codex 02-20-26.
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    mask = np.isfinite(df_all["flatness_ratio"]) & np.isfinite(df_all["trace_violation"])
    df_plot = df_all.loc[mask]

    sc = ax.scatter(
        df_plot["flatness_ratio"],
        df_plot["trace_violation"],
        c=df_plot["berry_std"],
        s=20,
        alpha=0.65,
        cmap="viridis",
    )
    ax.scatter(
        df_good["flatness_ratio"],
        df_good["trace_violation"],
        facecolors="none",
        edgecolors="black",
        s=56,
        linewidths=0.8,
        label=f"| |C| - 1 | <= {CHERN_MAG_TOL:.2f}",
    )

    ax.set_xlabel("Bandwidth / gap (lowest to second-lowest)")
    ax.set_ylabel("Trace condition violation: Q - |C|")
    ax.set_title(f"Broadened Anchor Search (lowest band, ph_conj=True, nk={EVAL_NK})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="Std[Berry curvature] in BZ")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_scatter_q_minus_absC_berry_std.png", dpi=180)
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
            f"C={float(row['chern_low']):.6f}, Q-|C|={float(row['trace_violation']):.6f}, "
            f"std(Berry)={float(row['berry_std']):.6f}; "
            f"(delta,t_hh1,t_th2,t_hh3,t_tt1)=({float(row['delta']):.4f},{float(row['t_hh1']):.4f},"
            f"{float(row['t_th2']):.4f},{float(row['t_hh3']):.4f},{float(row['t_tt1']):.4f})"
        ),
        fontsize=10,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _row_summary(row: pd.Series) -> dict[str, float]:
    # Written with Codex 02-20-26.
    keys = [
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
        "berry_std",
        "chern_mag_error",
    ]
    return {k: float(row[k]) for k in keys}


def main() -> None:
    # Written with Codex 02-20-26.
    out_dir = Path(__file__).resolve().parent

    print(f"Evaluating anchor baseline on nk={EVAL_NK} ...", flush=True)
    anchor_eval = _evaluate_samples(pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS), nk=EVAL_NK, gap_floor=GAP_FLOOR).iloc[0]
    print(
        f"Anchor: W/Delta={float(anchor_eval['flatness_ratio']):.6f}, "
        f"gap={float(anchor_eval['gap_low_mid']):.6f}, C={float(anchor_eval['chern_low']):.6f}, "
        f"Q-|C|={float(anchor_eval['trace_violation']):.6f}, std(Berry)={float(anchor_eval['berry_std']):.6f}",
        flush=True,
    )

    stage_tables: list[pd.DataFrame] = []
    per_seed_best: list[dict[str, float]] = []

    for seed in SEEDS:
        print(f"Seed round {seed}: stage 1 + stage 2 ...", flush=True)
        stage1, stage2 = _run_seed_round(
            seed=seed,
            anchor=ANCHOR,
            bounds=LOCAL_BOUNDS,
            nk_eval=EVAL_NK,
            gap_floor=GAP_FLOOR,
        )
        stage_tables.append(stage1)
        stage_tables.append(stage2)

        combo = pd.concat([stage1, stage2], ignore_index=True)
        combo = combo.loc[_candidate_mask(combo, chern_mag_tol=CHERN_MAG_TOL, gap_floor=GAP_FLOOR)].copy()
        combo["score"] = _score(combo)
        best_seed = combo.sort_values(["score", "flatness_ratio", "trace_violation", "berry_std"]).iloc[0]
        rec = _row_summary(best_seed)
        rec["round_seed"] = float(seed)
        per_seed_best.append(rec)

    all_eval = pd.concat(stage_tables, ignore_index=True)
    all_eval = _deduplicate_parameters(all_eval)

    print("Running final polish around best strict candidates ...", flush=True)
    polish = _polish_round(all_eval, bounds=LOCAL_BOUNDS, nk_eval=EVAL_NK, gap_floor=GAP_FLOOR)
    all_eval = pd.concat([all_eval, polish], ignore_index=True)
    all_eval = _deduplicate_parameters(all_eval)

    strict = all_eval.loc[_candidate_mask(all_eval, chern_mag_tol=CHERN_MAG_TOL, gap_floor=GAP_FLOOR)].copy()
    if len(strict) == 0:
        raise RuntimeError("No strict candidates survived after all rounds.")

    strict["score"] = _score(strict)
    strict = strict.sort_values(["score", "flatness_ratio", "trace_violation", "berry_std"]).reset_index(drop=True)

    best_by_flat = strict.sort_values(["flatness_ratio", "trace_violation", "berry_std", "chern_mag_error"]).head(1)
    best_by_trace = strict.sort_values(["trace_violation", "flatness_ratio", "berry_std", "chern_mag_error"]).head(1)
    best_by_berry_std = strict.sort_values(["berry_std", "flatness_ratio", "trace_violation", "chern_mag_error"]).head(1)
    top_candidates = strict.head(12).copy()

    all_eval.to_csv(out_dir / "all_candidates_evaluated.csv", index=False)
    strict.to_csv(out_dir / "strict_candidates_sorted.csv", index=False)
    top_candidates.to_csv(out_dir / "top_candidates.csv", index=False)
    best_by_flat.to_csv(out_dir / "best_by_flatness.csv", index=False)
    best_by_trace.to_csv(out_dir / "best_by_trace_violation.csv", index=False)
    best_by_berry_std.to_csv(out_dir / "best_by_berry_std.csv", index=False)
    pd.DataFrame(per_seed_best).to_csv(out_dir / "per_seed_best.csv", index=False)

    _plot_objective(df_all=all_eval, df_good=strict, out_dir=out_dir)

    selected = pd.concat([best_by_flat, best_by_trace, best_by_berry_std, strict.head(4)], ignore_index=True)
    selected = _deduplicate_parameters(selected).head(4).reset_index(drop=True)
    for i, row in selected.iterrows():
        _plot_candidate_band_and_geometry(
            row=row,
            out_path=out_dir / f"candidate_{i+1:02d}_band_and_quantum_geometry.png",
        )

    summary = {
        "seed_rounds": [int(s) for s in SEEDS],
        "search_type": "bestcheck_anchor_local_wider_with_berry_std",
        "ph_conj": True,
        "band": "lowest",
        "bandgap_definition": "E_band1 - E_band0",
        "objective": {
            "flatness": "bandwidth_low/gap_low_mid",
            "trace_violation": "quantum_weight_low - abs(chern_low)",
            "berry_std": "stddev of Berry curvature over sampled BZ",
        },
        "nk_evaluation": int(EVAL_NK),
        "gap_floor": float(GAP_FLOOR),
        "chern_mag_tolerance": float(CHERN_MAG_TOL),
        "local_bounds": {k: [float(v[0]), float(v[1])] for k, v in LOCAL_BOUNDS.items()},
        "counts": {
            "all_evaluated": int(len(all_eval)),
            "strict": int(len(strict)),
            "top_candidates": int(len(top_candidates)),
        },
        "anchor": _row_summary(anchor_eval),
        "best_by_flatness": _row_summary(best_by_flat.iloc[0]),
        "best_by_trace_violation": _row_summary(best_by_trace.iloc[0]),
        "best_by_berry_std": _row_summary(best_by_berry_std.iloc[0]),
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
        "berry_std",
    ]
    print("\nBest by flatness:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(best_by_flat[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nBest by trace violation:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(best_by_trace[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nBest by Berry-curvature std:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(best_by_berry_std[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nTop strict candidates:", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(top_candidates[cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)

    print("\nWrote files:", flush=True)
    for name in [
        "all_candidates_evaluated.csv",
        "strict_candidates_sorted.csv",
        "top_candidates.csv",
        "per_seed_best.csv",
        "best_by_flatness.csv",
        "best_by_trace_violation.csv",
        "best_by_berry_std.csv",
        "objective_scatter_q_minus_absC_berry_std.png",
        "candidate_01_band_and_quantum_geometry.png",
        "candidate_02_band_and_quantum_geometry.png",
        "candidate_03_band_and_quantum_geometry.png",
        "candidate_04_band_and_quantum_geometry.png",
        "summary.json",
        "script_used_for_search.py",
    ]:
        print(f"- {out_dir / name}", flush=True)


if __name__ == "__main__":
    main()
