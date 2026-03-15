from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_bloch_ed_bench")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aidan_custom.bloch_ed.haldane import build_haldane_projected_hamiltonian
from aidan_custom.bloch_ed.many_body import (
    _accumulate_many_body_entries,
    build_all_momentum_sector_bases,
    build_fock_basis,
    prepare_many_body_operator_data,
)
from aidan_custom.bloch_ed.workflow import (
    solve_projected_all_momentum_sectors,
    solve_projected_sector,
)

RESULT_DIR = Path(__file__).resolve().parent
RAW_DIR = RESULT_DIR / "raw_data"
PLOTS_DIR = RESULT_DIR / "plots"


def _haldane_projected(Lx: int, Ly: int, V1: float = 1.0):
    # Written with Codex 03-14-26.
    return build_haldane_projected_hamiltonian(
        Lx=int(Lx),
        Ly=int(Ly),
        t1=1.0,
        t2=float(-1.0 / (4.0 * np.cos(0.65))),
        phi=0.65,
        m=0.0,
        selected_bands=[0],
        V1=float(V1),
    )


def _basis_loop_with_index() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    dims = []
    for kx in range(4):
        for ky in range(4):
            basis = build_fock_basis(
                n_orbitals=projected.one_body.shape[0],
                n_particles=5,
                orbital_momenta=projected.orbital_momenta,
                lattice_shape=(4, 4),
                momentum_sector=(kx, ky),
                build_state_index=True,
            )
            dims.append(len(basis.states))
    return {"total_dim": float(sum(dims))}


def _basis_shared_no_index() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    bases = build_all_momentum_sector_bases(
        n_orbitals=projected.one_body.shape[0],
        n_particles=5,
        orbital_momenta=projected.orbital_momenta,
        lattice_shape=(4, 4),
        build_state_index=False,
    )
    return {
        "total_dim": float(sum(len(basis.states) for basis in bases.values())),
    }


def _basis_shared_with_index() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    bases = build_all_momentum_sector_bases(
        n_orbitals=projected.one_body.shape[0],
        n_particles=5,
        orbital_momenta=projected.orbital_momenta,
        lattice_shape=(4, 4),
        build_state_index=True,
    )
    return {
        "total_dim": float(sum(len(basis.states) for basis in bases.values())),
    }


def _dense_sector_legacy_python() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 3)
    basis = build_fock_basis(
        n_orbitals=projected.one_body.shape[0],
        n_particles=4,
        orbital_momenta=projected.orbital_momenta,
        lattice_shape=(4, 3),
        momentum_sector=(0, 0),
        build_state_index=True,
    )
    operator_data = prepare_many_body_operator_data(
        projected.one_body,
        projected.two_body,
        build_numba_compact=False,
    )
    rows, cols, data = _accumulate_many_body_entries(
        one_body=projected.one_body,
        two_body=projected.two_body,
        basis=basis,
        cutoff=1e-12,
        operator_data=operator_data,
    )
    h_dense = np.zeros((len(basis.states), len(basis.states)), dtype=np.complex128)
    for idx in range(data.size):
        h_dense[rows[idx], cols[idx]] += data[idx]
    h_dense = 0.5 * (h_dense + np.conjugate(h_dense.T))
    eigvals = np.linalg.eigvalsh(h_dense)
    return {"energy0": float(eigvals[0].real), "dim": float(len(basis.states))}


def _dense_sector_default() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 3)
    basis, _, eigvals, _ = solve_projected_sector(
        projected_hamiltonian=projected,
        n_particles=4,
        momentum_sector=(0, 0),
        n_eigs=1,
        use_sparse=False,
    )
    return {"energy0": float(eigvals[0].real), "dim": float(len(basis.states))}


def _dense_sector_with_index() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 3)
    basis = build_fock_basis(
        n_orbitals=projected.one_body.shape[0],
        n_particles=4,
        orbital_momenta=projected.orbital_momenta,
        lattice_shape=(4, 3),
        momentum_sector=(0, 0),
        build_state_index=True,
    )
    _, _, eigvals, _ = solve_projected_sector(
        projected_hamiltonian=projected,
        n_particles=4,
        momentum_sector=(0, 0),
        n_eigs=1,
        use_sparse=False,
        basis=basis,
    )
    return {"energy0": float(eigvals[0].real), "dim": float(len(basis.states))}


def _all_sector_sequential() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=5,
        n_eigs=1,
        use_sparse=True,
        parallel_sparse_sectors=False,
        sparse_threshold=200,
    )
    e0 = min(float(np.asarray(v["eigenvalues"])[0].real) for v in results.values())
    return {"energy0": e0, "n_sectors": float(len(results))}


def _all_sector_threaded() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=5,
        n_eigs=1,
        use_sparse=True,
        parallel_sparse_sectors=True,
        sparse_parallel_backend="thread",
        max_sector_workers=4,
        sparse_threshold=200,
    )
    e0 = min(float(np.asarray(v["eigenvalues"])[0].real) for v in results.values())
    return {"energy0": e0, "n_sectors": float(len(results))}


def _all_sector_process() -> dict[str, float]:
    # Written with Codex 03-14-26.
    projected = _haldane_projected(4, 4)
    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=5,
        n_eigs=1,
        use_sparse=True,
        parallel_sparse_sectors=True,
        sparse_parallel_backend="process",
        max_sector_workers=4,
        sparse_threshold=200,
    )
    e0 = min(float(np.asarray(v["eigenvalues"])[0].real) for v in results.values())
    return {"energy0": e0, "n_sectors": float(len(results))}


VARIANTS = {
    "basis_loop_with_index": _basis_loop_with_index,
    "basis_shared_no_index": _basis_shared_no_index,
    "basis_shared_with_index": _basis_shared_with_index,
    "dense_sector_legacy_python": _dense_sector_legacy_python,
    "dense_sector_default": _dense_sector_default,
    "dense_sector_with_index": _dense_sector_with_index,
    "all_sector_sequential": _all_sector_sequential,
    "all_sector_threaded": _all_sector_threaded,
    "all_sector_process": _all_sector_process,
}


def _run_variant(variant: str) -> dict[str, float]:
    # Written with Codex 03-14-26.
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}.")
    return VARIANTS[variant]()


def _child_main(variant: str) -> None:
    # Written with Codex 03-14-26.
    _run_variant(variant)
    t0 = time.perf_counter()
    payload = _run_variant(variant)
    elapsed = time.perf_counter() - t0
    rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        json.dumps(
            {
                "variant": variant,
                "seconds": float(elapsed),
                "maxrss_kib": rss_kib,
                "payload": payload,
            }
        )
    )


def _repeat_count(variant: str) -> int:
    # Written with Codex 03-14-26.
    if variant.startswith("basis_"):
        return 5
    if variant.startswith("dense_sector_"):
        return 4
    return 3


def _run_child_variant(variant: str) -> list[dict[str, object]]:
    # Written with Codex 03-14-26.
    rows: list[dict[str, object]] = []
    for repeat in range(_repeat_count(variant)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", variant],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        rows.append(
            {
                "variant": variant,
                "repeat": int(repeat),
                "seconds": float(payload["seconds"]),
                "maxrss_kib": float(payload["maxrss_kib"]),
                "payload": payload["payload"],
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    # Written with Codex 03-14-26.
    header = "variant,repeat,seconds,maxrss_kib\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for row in rows:
            f.write(
                f"{row['variant']},{row['repeat']},{row['seconds']},{row['maxrss_kib']}\n"
            )


def _plot_metric(rows: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    # Written with Codex 03-14-26.
    variants = sorted({str(row["variant"]) for row in rows})
    means = np.asarray(
        [
            np.mean([float(row[key]) for row in rows if str(row["variant"]) == variant])
            for variant in variants
        ],
        dtype=np.float64,
    )
    stds = np.asarray(
        [
            np.std(
                [float(row[key]) for row in rows if str(row["variant"]) == variant],
                ddof=1,
            )
            if sum(str(row["variant"]) == variant for row in rows) > 1
            else 0.0
            for variant in variants
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(variants), dtype=np.int64)
    ax.bar(x, means, yerr=stds, capsize=5.0, color="#4c7dad")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(rows: list[dict[str, object]], path: Path) -> None:
    # Written with Codex 03-14-26.
    variants = sorted({str(row["variant"]) for row in rows})
    summary = {}
    for variant in variants:
        subset = [row for row in rows if str(row["variant"]) == variant]
        summary[variant] = {
            "seconds_mean": float(np.mean([float(row["seconds"]) for row in subset])),
            "seconds_std": float(np.std([float(row["seconds"]) for row in subset], ddof=1))
            if len(subset) > 1
            else 0.0,
            "maxrss_kib_mean": float(
                np.mean([float(row["maxrss_kib"]) for row in subset])
            ),
            "payload": subset[-1]["payload"],
        }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    # Written with Codex 03-14-26.
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=str, default=None)
    args = parser.parse_args()

    if args.child is not None:
        _child_main(args.child)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows.extend(_run_child_variant(variant))

    _write_csv(rows, RAW_DIR / "benchmark_rows.csv")
    _write_summary(rows, RAW_DIR / "summary.json")
    _plot_metric(rows, "seconds", "Wall time (s)", PLOTS_DIR / "timings.png")
    _plot_metric(rows, "maxrss_kib", "Peak RSS (KiB)", PLOTS_DIR / "maxrss.png")


if __name__ == "__main__":
    main()
