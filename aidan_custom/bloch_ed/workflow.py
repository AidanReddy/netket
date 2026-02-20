from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

from .many_body import (
    build_fock_basis,
    build_many_body_hamiltonian_dense,
    build_many_body_hamiltonian_sparse,
    diagonalize_many_body_hamiltonian,
)
from .types import ManyBodyBasis, ProjectedFermionHamiltonian


def solve_projected_sector(
    projected_hamiltonian: ProjectedFermionHamiltonian,
    n_particles: int,
    momentum_sector: tuple[int, int] | None = None,
    cutoff: float = 1e-12,
    n_eigs: int = 1,
    sparse_threshold: int = 2000,
    use_sparse: bool = True,
) -> tuple[ManyBodyBasis, np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray]:
    # Written with Codex 02-19-26.
    lattice = projected_hamiltonian.lattice
    n_orbitals = projected_hamiltonian.one_body.shape[0]
    basis = build_fock_basis(
        n_orbitals=n_orbitals,
        n_particles=n_particles,
        orbital_momenta=projected_hamiltonian.orbital_momenta,
        lattice_shape=(lattice.Lx, lattice.Ly),
        momentum_sector=momentum_sector,
    )
    if len(basis.states) == 0:
        empty_h = (
            sp.csr_matrix((0, 0), dtype=np.complex128)
            if use_sparse
            else np.zeros((0, 0), dtype=np.complex128)
        )
        return (
            basis,
            empty_h,
            np.asarray([], dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )

    if use_sparse:
        h_many = build_many_body_hamiltonian_sparse(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=cutoff,
        )
    else:
        h_many = build_many_body_hamiltonian_dense(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=cutoff,
        )

    n_eigs_eff = min(int(n_eigs), len(basis.states))
    eigvals, eigvecs = diagonalize_many_body_hamiltonian(
        hamiltonian=h_many,
        n_eigs=n_eigs_eff,
        sparse_threshold=sparse_threshold,
    )
    return basis, h_many, eigvals, eigvecs


def solve_projected_all_momentum_sectors(
    projected_hamiltonian: ProjectedFermionHamiltonian,
    n_particles: int,
    cutoff: float = 1e-12,
    n_eigs: int = 1,
    sparse_threshold: int = 1000,
    use_sparse: bool = True,
    n_eigs_sparse: int | None = None,
) -> Mapping[tuple[int, int], dict[str, object]]:
    # Written with Codex 02-19-26.
    lattice = projected_hamiltonian.lattice
    results: dict[tuple[int, int], dict[str, object]] = {}

    for kx in range(lattice.Lx):
        for ky in range(lattice.Ly):
            sector = (kx, ky)
            basis_preview = build_fock_basis(
                n_orbitals=projected_hamiltonian.one_body.shape[0],
                n_particles=n_particles,
                orbital_momenta=projected_hamiltonian.orbital_momenta,
                lattice_shape=(lattice.Lx, lattice.Ly),
                momentum_sector=sector,
            )
            dim = len(basis_preview.states)
            if dim == 0:
                continue

            use_sparse_sector = bool(use_sparse) and (dim > sparse_threshold)
            if use_sparse_sector and n_eigs_sparse is not None:
                n_eigs_sector = int(n_eigs_sparse)
            else:
                n_eigs_sector = int(n_eigs)

            basis, h_many, eigvals, eigvecs = solve_projected_sector(
                projected_hamiltonian=projected_hamiltonian,
                n_particles=n_particles,
                momentum_sector=sector,
                cutoff=cutoff,
                n_eigs=n_eigs_sector,
                sparse_threshold=sparse_threshold,
                use_sparse=use_sparse_sector,
            )
            results[sector] = {
                "basis": basis,
                "hamiltonian": h_many,
                "eigenvalues": eigvals,
                "eigenvectors": eigvecs,
                "solver": "sparse" if use_sparse_sector else "dense",
            }

    return results
