from __future__ import annotations

import json
import shutil
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

PARAM_COLUMNS = ["delta", "t_hh1", "t_th2", "t_hh3", "t_tt1"]
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "delta": (2.0, 8.0),
    "t_hh1": (0.05, 1.20),
    "t_th2": (-0.35, 0.35),
    "t_hh3": (-0.35, 0.35),
    "t_tt1": (-0.50, 0.50),
}
ANCHOR = {
    "delta": 27.9 / 5.15,
    "t_hh1": 1.81 / 5.15,
    "t_th2": 0.07 / 5.15,
    "t_hh3": 0.43 / 5.15,
    "t_tt1": -0.46 / 5.15,
}
GAP_FLOOR = 5.0e-4


def _sample_uniform(rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    n_samples = int(max(0, n_samples))
    data = {
        name: rng.uniform(low, high, size=n_samples)
        for name, (low, high) in PARAM_RANGES.items()
    }
    return pd.DataFrame(data, columns=PARAM_COLUMNS)


def _clip_parameters(df: pd.DataFrame) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    out = df.copy()
    for name, (low, high) in PARAM_RANGES.items():
        out[name] = np.clip(out[name].to_numpy(float), low, high)
    return out


def _sample_local(
    rng: np.random.Generator,
    seeds: pd.DataFrame,
    n_per_seed: int,
    scales: dict[str, float],
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    n_per_seed = int(max(0, n_per_seed))
    if n_per_seed == 0 or len(seeds) == 0:
        return pd.DataFrame(columns=PARAM_COLUMNS)

    seed_vals = seeds[PARAM_COLUMNS].to_numpy(float)
    rows: list[dict[str, float]] = []
    for row in seed_vals:
        for _ in range(n_per_seed):
            sample = {}
            for j, name in enumerate(PARAM_COLUMNS):
                sample[name] = float(row[j] + rng.normal(0.0, scales[name]))
            rows.append(sample)

    if not rows:
        return pd.DataFrame(columns=PARAM_COLUMNS)
    return _clip_parameters(pd.DataFrame(rows, columns=PARAM_COLUMNS))


def _deduplicate_parameters(df: pd.DataFrame, decimals: int = 8) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    rounded = df[PARAM_COLUMNS].round(decimals)
    keep = ~rounded.duplicated()
    return df.loc[keep].reset_index(drop=True)


def _k_mesh(nk: int, a_m: float = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=a_m)
    k_points = sample_shortest_representative_bz(nk, nk, b1, b2, unique=False)
    area_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    prefactor = area_bz / (2.0 * np.pi * nk * nk)
    return k_points, prefactor, b1, b2


def _staggered_gap_probe_points(nk: int, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    # Written with Codex 02-20-26.
    nk = int(nk)
    coeff = (np.arange(nk, dtype=float) + 0.5) / float(nk)
    a, b = np.meshgrid(coeff, coeff, indexing="ij")
    probe = np.empty((nk * nk, 2), dtype=float)
    probe[:, 0] = a.ravel() * float(b1[0]) + b.ravel() * float(b2[0])
    probe[:, 1] = a.ravel() * float(b1[1]) + b.ravel() * float(b2[1])
    return probe


def _evaluate_samples(param_df: pd.DataFrame, nk: int) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    k_points, prefactor, b1, b2 = _k_mesh(nk=nk, a_m=1.0)
    k_gap_probe = _staggered_gap_probe_points(nk=nk, b1=b1, b2=b2)
    k_gap_all = np.concatenate([k_points, k_gap_probe], axis=0)

    n = len(param_df)
    bandwidth = np.full(n, np.nan, dtype=float)
    min_gap = np.full(n, np.nan, dtype=float)
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
        gap_min = np.inf

        # First pass: robustly check bandwidth and direct gap on regular + staggered meshes.
        for kx, ky in k_gap_all:
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
            if gap < gap_min:
                gap_min = gap

        bw = e_max - e_min
        bandwidth[i] = bw
        min_gap[i] = gap_min
        if gap_min <= GAP_FLOOR:
            flatness[i] = np.inf
            continue

        flatness[i] = bw / gap_min

        # Second pass: integrate geometry on the regular nk x nk mesh.
        berry_sum = 0.0
        metric_sum = 0.0
        failed = False
        for kx, ky in k_points:
            try:
                berry, metric = geo_fn(
                    kx,
                    ky,
                    "lowest",
                    1.0e-5,
                    GAP_FLOOR,
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
    out["gap_low_mid"] = min_gap
    out["flatness_ratio"] = flatness
    out["chern_low"] = chern
    out["quantum_weight_low"] = qweight
    out["trace_violation"] = qweight - chern
    out["trace_violation_abs"] = np.abs(out["trace_violation"])
    out["chern_error"] = np.abs(chern - 1.0)
    out["is_gapped"] = min_gap > GAP_FLOOR
    return out


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


def _candidate_subset(df: pd.DataFrame, chern_tol: float) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    mask = (
        df["is_gapped"].to_numpy(bool)
        & np.isfinite(df["flatness_ratio"].to_numpy(float))
        & np.isfinite(df["trace_violation"].to_numpy(float))
        & (np.abs(df["chern_low"].to_numpy(float) - 1.0) <= float(chern_tol))
    )
    return df.loc[mask].copy()


def _score_candidates(df: pd.DataFrame) -> np.ndarray:
    # Written with Codex 02-20-26.
    flat_norm = _normalized(df["flatness_ratio"].to_numpy(float))
    trace_norm = _normalized(df["trace_violation_abs"].to_numpy(float))
    chern_norm = _normalized(df["chern_error"].to_numpy(float))
    return flat_norm + trace_norm + 2.0 * chern_norm


def _select_seeds(
    df: pd.DataFrame,
    n_seeds: int,
    chern_tol: float,
    relaxed_tol: float,
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    n_seeds = int(max(1, n_seeds))

    candidates = _candidate_subset(df, chern_tol=chern_tol)
    if len(candidates) < max(8, n_seeds // 3):
        candidates = _candidate_subset(df, chern_tol=relaxed_tol)

    if len(candidates) < max(8, n_seeds // 3):
        valid_mask = (
            df["is_gapped"].to_numpy(bool)
            & np.isfinite(df["flatness_ratio"].to_numpy(float))
            & np.isfinite(df["trace_violation"].to_numpy(float))
        )
        candidates = df.loc[valid_mask].copy()

    if len(candidates) == 0:
        raise RuntimeError("No valid candidates found for seed selection.")

    candidates = candidates.copy()
    candidates["stage_score"] = _score_candidates(candidates)
    ordered = candidates.sort_values(
        ["stage_score", "chern_error", "flatness_ratio", "trace_violation_abs"]
    )
    return ordered.head(n_seeds)[PARAM_COLUMNS].reset_index(drop=True)


def _make_stage_parameters(
    rng: np.random.Generator,
    seeds: pd.DataFrame,
    n_local_per_seed: int,
    n_global_random: int,
    scales: dict[str, float],
) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    local_df = _sample_local(
        rng=rng,
        seeds=seeds,
        n_per_seed=n_local_per_seed,
        scales=scales,
    )
    global_df = _sample_uniform(rng=rng, n_samples=n_global_random)

    pieces = [pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS), seeds[PARAM_COLUMNS], local_df, global_df]
    merged = pd.concat(pieces, ignore_index=True)
    merged = _clip_parameters(merged)
    return _deduplicate_parameters(merged)


def _best_rows(df: pd.DataFrame, chern_tol: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Written with Codex 02-20-26.
    c1 = _candidate_subset(df, chern_tol=chern_tol)
    if len(c1) == 0:
        raise RuntimeError(
            f"No points found with |C-1| <= {chern_tol:.3f} in the final stage."
        )

    best_flat = c1.sort_values(["flatness_ratio", "trace_violation_abs", "chern_error"]).head(1)
    best_trace = c1.sort_values(["trace_violation", "flatness_ratio", "chern_error"]).head(1)
    return c1, best_flat, best_trace


def _make_plots(
    df_final: pd.DataFrame,
    df_c1: pd.DataFrame,
    best_flat: pd.DataFrame,
    best_trace: pd.DataFrame,
    out_dir: Path,
) -> None:
    # Written with Codex 02-20-26.
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    all_mask = np.isfinite(df_final["flatness_ratio"]) & np.isfinite(df_final["trace_violation"])
    df_plot = df_final.loc[all_mask]

    sc = ax.scatter(
        df_plot["flatness_ratio"],
        df_plot["trace_violation"],
        c=df_plot["chern_low"],
        cmap="coolwarm",
        s=24,
        alpha=0.75,
    )
    ax.scatter(
        df_c1["flatness_ratio"],
        df_c1["trace_violation"],
        facecolors="none",
        edgecolors="black",
        s=62,
        linewidths=0.8,
        label="|C-1| window",
    )

    row_flat = best_flat.iloc[0]
    row_trace = best_trace.iloc[0]
    ax.scatter(
        [row_flat["flatness_ratio"]],
        [row_flat["trace_violation"]],
        marker="*",
        color="gold",
        s=220,
        edgecolors="black",
        linewidths=0.8,
        label="Best flatness",
    )
    ax.scatter(
        [row_trace["flatness_ratio"]],
        [row_trace["trace_violation"]],
        marker="D",
        color="limegreen",
        s=80,
        edgecolors="black",
        linewidths=0.8,
        label="Best trace violation",
    )

    ax.set_xlabel("Bandwidth / bandgap (lowest band)")
    ax.set_ylabel("Trace condition violation: Q - C")
    ax.set_title("MoTe2 three-orbital search (lowest, ph_conj=True)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="Chern number (lowest band)")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_scatter.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    sc0 = axes[0].scatter(
        df_c1["delta"],
        df_c1["t_hh1"],
        c=df_c1["flatness_ratio"],
        s=35,
        cmap="viridis",
    )
    axes[0].set_xlabel("delta")
    axes[0].set_ylabel("t_hh1")
    axes[0].set_title("C~1 candidates: flatness")
    axes[0].grid(alpha=0.25)
    fig.colorbar(sc0, ax=axes[0], label="W / Delta")

    sc1 = axes[1].scatter(
        df_c1["t_th2"],
        df_c1["t_hh3"],
        c=df_c1["trace_violation"],
        s=35,
        cmap="magma",
    )
    axes[1].set_xlabel("t_th2")
    axes[1].set_ylabel("t_hh3")
    axes[1].set_title("C~1 candidates: Q - C")
    axes[1].grid(alpha=0.25)
    fig.colorbar(sc1, ax=axes[1], label="Q - C")

    fig.savefig(out_dir / "parameter_scatter_c1.png", dpi=180)
    plt.close(fig)


def _row_to_dict(row: pd.Series) -> dict[str, float]:
    # Written with Codex 02-20-26.
    out: dict[str, float] = {}
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
        "chern_error",
        "nk",
    ]:
        out[key] = float(row[key])
    return out


def main() -> None:
    # Written with Codex 02-20-26.
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(20260220)

    n_coarse = 1400
    n_medium_seed = 60
    n_final_seed = 45
    nk_coarse = 9
    nk_medium = 21
    nk_final = 21
    chern_tol_final = 0.10

    print(f"Stage 1: coarse global search (gap floor={GAP_FLOOR:.1e})", flush=True)
    coarse_params = pd.concat(
        [
            pd.DataFrame([ANCHOR], columns=PARAM_COLUMNS),
            _sample_uniform(rng=rng, n_samples=n_coarse),
        ],
        ignore_index=True,
    )
    coarse_params = _deduplicate_parameters(_clip_parameters(coarse_params))
    df_coarse = _evaluate_samples(coarse_params, nk=nk_coarse)

    print("Stage 2: medium refinement", flush=True)
    seeds_medium = _select_seeds(
        df=df_coarse,
        n_seeds=n_medium_seed,
        chern_tol=0.35,
        relaxed_tol=0.70,
    )
    medium_params = _make_stage_parameters(
        rng=rng,
        seeds=seeds_medium,
        n_local_per_seed=3,
        n_global_random=170,
        scales={
            "delta": 0.30,
            "t_hh1": 0.06,
            "t_th2": 0.05,
            "t_hh3": 0.05,
            "t_tt1": 0.07,
        },
    )
    df_medium = _evaluate_samples(medium_params, nk=nk_medium)

    print("Stage 3: final refinement", flush=True)
    seeds_final = _select_seeds(
        df=df_medium,
        n_seeds=n_final_seed,
        chern_tol=0.20,
        relaxed_tol=0.40,
    )
    final_params = _make_stage_parameters(
        rng=rng,
        seeds=seeds_final,
        n_local_per_seed=2,
        n_global_random=70,
        scales={
            "delta": 0.14,
            "t_hh1": 0.030,
            "t_th2": 0.025,
            "t_hh3": 0.025,
            "t_tt1": 0.035,
        },
    )
    df_final = _evaluate_samples(final_params, nk=nk_final)

    c1_final, best_flat, best_trace = _best_rows(df_final, chern_tol=chern_tol_final)

    if len(c1_final) < 3:
        c1_final, best_flat, best_trace = _best_rows(df_final, chern_tol=0.15)
        chern_tol_final = 0.15

    best_flat.to_csv(out_dir / "best_by_flatness.csv", index=False)
    best_trace.to_csv(out_dir / "best_by_trace_violation.csv", index=False)

    df_coarse.to_csv(out_dir / "coarse_stage.csv", index=False)
    df_medium.to_csv(out_dir / "medium_stage.csv", index=False)
    df_final.to_csv(out_dir / "final_stage.csv", index=False)
    c1_final.sort_values(["flatness_ratio", "trace_violation_abs"]).to_csv(
        out_dir / "chern1_candidates_final.csv", index=False
    )

    _make_plots(
        df_final=df_final,
        df_c1=c1_final,
        best_flat=best_flat,
        best_trace=best_trace,
        out_dir=out_dir,
    )

    summary = {
        "seed": 20260220,
        "ph_conj": True,
        "band": "lowest",
        "chern_target": 1.0,
        "chern_tolerance_used": float(chern_tol_final),
        "gap_floor": float(GAP_FLOOR),
        "n_evaluated": {
            "coarse": int(len(df_coarse)),
            "medium": int(len(df_medium)),
            "final": int(len(df_final)),
        },
        "n_chern1_final": int(len(c1_final)),
        "best_by_flatness": _row_to_dict(best_flat.iloc[0]),
        "best_by_trace_violation": _row_to_dict(best_trace.iloc[0]),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    shutil.copyfile(__file__, out_dir / "script_used_for_plots.py")

    cols = [
        "delta",
        "t_hh1",
        "t_th2",
        "t_hh3",
        "t_tt1",
        "flatness_ratio",
        "trace_violation",
        "chern_low",
        "quantum_weight_low",
        "gap_low_mid",
        "bandwidth_low",
        "chern_error",
        "nk",
    ]
    print("\nBest by flatness (C~1 subset):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(best_flat[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    print("\nBest by trace violation (C~1 subset):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(best_trace[cols].to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    print("\nWrote files:")
    for name in [
        "coarse_stage.csv",
        "medium_stage.csv",
        "final_stage.csv",
        "chern1_candidates_final.csv",
        "best_by_flatness.csv",
        "best_by_trace_violation.csv",
        "objective_scatter.png",
        "parameter_scatter_c1.png",
        "summary.json",
        "script_used_for_plots.py",
    ]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
