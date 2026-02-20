from __future__ import annotations

import os

# Written with Codex 02-19-26.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from .haldane import build_haldane_projected_hamiltonian
from .workflow import solve_projected_all_momentum_sectors


def main() -> None:
    # Written with Codex 02-19-26.
    projected = build_haldane_projected_hamiltonian(
        Lx=3,
        Ly=3,
        t1=1.0,
        t2=-1.0 / (4.0 * np.cos(0.65)),
        phi=0.65,
        m=0.0,
        selected_bands=[0],
        V1=1.0,
    )

    results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=3,
        n_eigs=1,
        use_sparse=True,
    )

    for sector in sorted(results):
        basis = results[sector]["basis"]
        energy = float(results[sector]["eigenvalues"][0])
        print(f"sector={sector} dim={len(basis.states)} E0={energy:.12f}")


if __name__ == "__main__":
    main()
