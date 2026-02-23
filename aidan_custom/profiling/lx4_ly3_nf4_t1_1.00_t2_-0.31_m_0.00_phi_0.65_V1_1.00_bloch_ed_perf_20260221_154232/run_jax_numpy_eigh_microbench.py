from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp

RESULT_DIR = Path(__file__).resolve().parent
RAW_DIR = RESULT_DIR / "raw_data"
PLOTS_DIR = RESULT_DIR / "plots"


def benchmark_single_size(n: int, repeats: int, rng: np.random.Generator) -> dict[str, float]:
    # Written with Codex 02-21-26.
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (x + x.conjugate().T)

    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        np.linalg.eigh(h)
    numpy_seconds = (time.perf_counter() - t0) / float(repeats)

    hj = jnp.asarray(h)
    eig_fn = jax.jit(lambda m: jnp.linalg.eigh(m)[0])
    eig_fn(hj).block_until_ready()
    t1 = time.perf_counter()
    for _ in range(int(repeats)):
        eig_fn(hj).block_until_ready()
    jax_seconds = (time.perf_counter() - t1) / float(repeats)

    return {
        "matrix_dim": float(n),
        "numpy_seconds": float(numpy_seconds),
        "jax_seconds": float(jax_seconds),
        "numpy_over_jax_ratio": float(numpy_seconds / jax_seconds),
    }


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    # Written with Codex 02-21-26.
    fieldnames = [
        "matrix_dim",
        "numpy_seconds",
        "jax_seconds",
        "numpy_over_jax_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float]], output_path: Path) -> None:
    # Written with Codex 02-21-26.
    dims = np.asarray([int(row["matrix_dim"]) for row in rows], dtype=np.int64)
    numpy_t = np.asarray([row["numpy_seconds"] for row in rows], dtype=np.float64)
    jax_t = np.asarray([row["jax_seconds"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(dims, numpy_t, "o-", lw=1.5, label="numpy.linalg.eigh")
    ax.plot(dims, jax_t, "s-", lw=1.5, label="jax.jit(jnp.linalg.eigh)")
    ax.set_xlabel("Matrix dimension")
    ax.set_ylabel("Seconds per call")
    ax.set_title("Dense eigh microbenchmark (CPU backend)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Written with Codex 02-21-26.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260221)

    sizes = [24, 32, 48, 64, 128, 256]
    repeats = 20
    rows = [
        benchmark_single_size(n=size, repeats=repeats, rng=rng)
        for size in sizes
    ]
    write_csv(rows=rows, output_path=RAW_DIR / "jax_numpy_eigh_cpu_microbench.csv")
    plot_rows(rows=rows, output_path=PLOTS_DIR / "jax_numpy_eigh_cpu_microbench.png")

    print("backend", jax.default_backend())
    for row in rows:
        print(
            f"n={int(row['matrix_dim'])} "
            f"numpy={row['numpy_seconds']:.6f}s "
            f"jax={row['jax_seconds']:.6f}s "
            f"np_over_jax={row['numpy_over_jax_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
