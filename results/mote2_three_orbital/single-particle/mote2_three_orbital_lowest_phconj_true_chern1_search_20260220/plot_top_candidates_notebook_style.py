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


def _select_top_candidates(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Written with Codex 02-20-26.
    out = df.copy()
    flat_norm = _normalized(out["flatness_ratio"].to_numpy(float))
    trace_norm = _normalized(out["trace_violation_abs"].to_numpy(float))
    chern_norm = _normalized(out["chern_error"].to_numpy(float))
    out["leader_score"] = flat_norm + trace_norm + 0.5 * chern_norm
    return out.sort_values(["leader_score", "flatness_ratio", "trace_violation_abs"]).head(top_n).reset_index(drop=True)


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


def _integrated_geometry(params: dict[str, float], nx: int, ny: int) -> tuple[float, float]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=params["a_m"])
    k_points = sample_shortest_representative_bz(nx, ny, b1, b2, unique=False)

    berry_sum = 0.0
    metric_sum = 0.0
    for kx, ky in k_points:
        berry, metric = mote2_three_orbital_berry_and_metric_trace(
            kx=kx,
            ky=ky,
            band="lowest",
            ph_conj=True,
            **params,
        )
        berry_sum += berry
        metric_sum += metric

    area_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    prefactor = area_bz / (2.0 * np.pi * nx * ny)
    return prefactor * berry_sum, prefactor * metric_sum


def _scatter_geometry(params: dict[str, float], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=params["a_m"])
    gamma, k_node, m_node, kp_node = high_symmetry_points(b1, b2)
    bz_hex = first_bz_hexagon_vertices(b1, b2)

    k_points = sample_shortest_representative_bz(nx, ny, b1, b2, unique=True)
    berry = np.empty(k_points.shape[0], dtype=float)
    metric_trace = np.empty(k_points.shape[0], dtype=float)

    for i, (kx, ky) in enumerate(k_points):
        berry_i, metric_i = mote2_three_orbital_berry_and_metric_trace(
            kx=kx,
            ky=ky,
            band="lowest",
            ph_conj=True,
            **params,
        )
        berry[i] = berry_i
        metric_trace[i] = metric_i

    special = np.array([gamma, k_node, m_node, kp_node])
    return k_points, berry, metric_trace, bz_hex, special


def _band_structure(params: dict[str, float], n_path: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=params["a_m"])
    k_path, x, x_nodes, labels = k_path_gamma_k_m_kp_gamma(n_path, b1, b2, fold=True)
    bands = np.array(
        [mote2_three_orbital_eigenvalues(kx, ky, ph_conj=True, **params) for kx, ky in k_path],
        dtype=float,
    )
    return bands, x, x_nodes, labels


def _plot_candidate(
    row: pd.Series,
    out_path: Path,
    n_path: int,
    nx_geo: int,
    ny_geo: int,
) -> dict[str, float]:
    # Written with Codex 02-20-26.
    params = _candidate_params(row)
    bands, x, x_nodes, labels = _band_structure(params=params, n_path=n_path)
    k_points, berry, metric_trace, bz_hex, special = _scatter_geometry(
        params=params,
        nx=nx_geo,
        ny=ny_geo,
    )
    chern_int, qweight_int = _integrated_geometry(params=params, nx=nx_geo, ny=ny_geo)

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
    ax.set_title("Band structure (ph_conj=True)")
    ax.grid(axis="y", alpha=0.2)

    hex_x = np.r_[bz_hex[:, 0], bz_hex[0, 0]]
    hex_y = np.r_[bz_hex[:, 1], bz_hex[0, 1]]

    ax = axes[1]
    berry_lim = float(np.max(np.abs(berry)))
    berry_lim = max(berry_lim, 1.0e-12)
    sc0 = ax.scatter(
        k_points[:, 0],
        k_points[:, 1],
        c=berry,
        s=20,
        cmap="jet_r",
        vmin=-berry_lim,
        vmax=berry_lim,
    )
    ax.plot(hex_x, hex_y, color="black", lw=1.0, zorder=-1)
    ax.scatter(special[:, 0], special[:, 1], color="black", s=16)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("Berry curvature")
    fig.colorbar(sc0, ax=ax, shrink=0.84)

    ax = axes[2]
    metric_max = float(np.max(metric_trace))
    metric_max = max(metric_max, 1.0e-12)
    sc1 = ax.scatter(
        k_points[:, 0],
        k_points[:, 1],
        c=metric_trace,
        s=20,
        cmap="jet",
        vmin=0.0,
        vmax=metric_max,
    )
    ax.plot(hex_x, hex_y, color="black", lw=1.0, zorder=-1)
    ax.scatter(special[:, 0], special[:, 1], color="black", s=16)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("Metric trace tr(g)")
    fig.colorbar(sc1, ax=ax, shrink=0.84)

    title = (
        f"Candidate {int(row['rank'])}: C={chern_int:.6f}, Q={qweight_int:.6f}, Q-C={qweight_int - chern_int:.6f}; "
        f"W/Delta={float(row['flatness_ratio']):.6f}, "
        f"(delta,t_hh1,t_th2,t_hh3,t_tt1)=({float(row['delta']):.4f},{float(row['t_hh1']):.4f},"
        f"{float(row['t_th2']):.4f},{float(row['t_hh3']):.4f},{float(row['t_tt1']):.4f})"
    )
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "rank": float(row["rank"]),
        "delta": float(row["delta"]),
        "t_hh1": float(row["t_hh1"]),
        "t_th2": float(row["t_th2"]),
        "t_hh3": float(row["t_hh3"]),
        "t_tt1": float(row["t_tt1"]),
        "flatness_ratio": float(row["flatness_ratio"]),
        "trace_violation_stage": float(row["trace_violation"]),
        "chern_stage": float(row["chern_low"]),
        "chern_integrated_plot_mesh": float(chern_int),
        "quantum_weight_integrated_plot_mesh": float(qweight_int),
        "trace_violation_integrated_plot_mesh": float(qweight_int - chern_int),
    }


def main() -> None:
    # Written with Codex 02-20-26.
    out_dir = Path(__file__).resolve().parent
    input_csv = out_dir / "final_stage.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing candidate file: {input_csv}")

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        chern_tol = float(summary.get("chern_tolerance_used", 0.10))
    else:
        chern_tol = 0.10
    top_n = 4
    n_path = 180
    nx_geo = 51
    ny_geo = 51

    df = pd.read_csv(input_csv)
    mask = (
        df["is_gapped"].to_numpy(bool)
        & np.isfinite(df["flatness_ratio"].to_numpy(float))
        & np.isfinite(df["trace_violation"].to_numpy(float))
        & (np.abs(df["chern_low"].to_numpy(float) - 1.0) <= chern_tol)
    )
    df_chern1 = df.loc[mask].copy()
    if len(df_chern1) == 0:
        raise RuntimeError(f"No final-stage points satisfy |C-1| <= {chern_tol:.3f}.")

    selected = _select_top_candidates(df=df_chern1, top_n=min(top_n, len(df_chern1))).copy()
    selected.insert(0, "rank", np.arange(1, len(selected) + 1, dtype=int))

    records: list[dict[str, float]] = []
    for _, row in selected.iterrows():
        rank = int(row["rank"])
        out_png = out_dir / f"candidate_{rank:02d}_band_and_quantum_geometry.png"
        record = _plot_candidate(
            row=row,
            out_path=out_png,
            n_path=n_path,
            nx_geo=nx_geo,
            ny_geo=ny_geo,
        )
        records.append(record)
        print(f"Wrote {out_png}")

    summary_df = pd.DataFrame.from_records(records)
    summary_df.to_csv(out_dir / "top_candidates_plots_summary_chern1_strict.csv", index=False)

    selected.to_csv(out_dir / "top_candidates_selected_chern1_strict.csv", index=False)
    shutil.copyfile(__file__, out_dir / "script_used_for_candidate_plots.py")

    print("Wrote summary tables:")
    print(f"- {out_dir / 'top_candidates_selected_chern1_strict.csv'}")
    print(f"- {out_dir / 'top_candidates_plots_summary_chern1_strict.csv'}")
    print(f"- {out_dir / 'script_used_for_candidate_plots.py'}")


if __name__ == "__main__":
    main()
