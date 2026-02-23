from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

from .many_body import (
    ManyBodyOperatorData,
    build_fock_basis,
    build_many_body_hamiltonian_dense,
    build_many_body_hamiltonian_sparse,
    diagonalize_many_body_hamiltonian,
    prepare_many_body_operator_data,
)
from .types import ManyBodyBasis, ProjectedFermionHamiltonian

_PROCESS_SOLVE_CONTEXT: tuple[
    ProjectedFermionHamiltonian,
    int,
    float,
    int,
    ManyBodyOperatorData,
] | None = None


def _sector_worker_count(
    n_tasks: int,
    max_sector_workers: int | None,
) -> int:
    # Written with Codex 02-21-26.
    n_tasks_int = int(max(0, n_tasks))
    if n_tasks_int == 0:
        return 0
    if max_sector_workers is not None:
        return max(1, min(int(max_sector_workers), n_tasks_int))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_tasks_int))


def _solve_sector_from_precomputed_basis(
    projected_hamiltonian: ProjectedFermionHamiltonian,
    n_particles: int,
    cutoff: float,
    n_eigs_sector: int,
    sparse_threshold: int,
    use_sparse_sector: bool,
    basis: ManyBodyBasis,
    operator_data: ManyBodyOperatorData,
) -> tuple[ManyBodyBasis, np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    return solve_projected_sector(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=n_particles,
        momentum_sector=basis.momentum_sector,
        cutoff=cutoff,
        n_eigs=n_eigs_sector,
        sparse_threshold=sparse_threshold,
        use_sparse=use_sparse_sector,
        basis=basis,
        operator_data=operator_data,
    )


def _normalize_sparse_parallel_backend(backend: str) -> str:
    # Written with Codex 02-21-26.
    value = str(backend).strip().lower()
    if value not in {"thread", "process"}:
        raise ValueError(
            "sparse_parallel_backend must be either 'thread' or 'process', "
            f"got {backend!r}."
        )
    return value


def _init_sparse_sector_process_context(
    projected_hamiltonian: ProjectedFermionHamiltonian,
    n_particles: int,
    cutoff: float,
    sparse_threshold: int,
    operator_data: ManyBodyOperatorData,
) -> None:
    # Written with Codex 02-21-26.
    global _PROCESS_SOLVE_CONTEXT
    _PROCESS_SOLVE_CONTEXT = (
        projected_hamiltonian,
        int(n_particles),
        float(cutoff),
        int(sparse_threshold),
        operator_data,
    )


def _solve_sparse_sector_process_task(
    task: tuple[tuple[int, int], ManyBodyBasis, int, bool],
) -> tuple[tuple[int, int], np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    if _PROCESS_SOLVE_CONTEXT is None:
        raise RuntimeError("Sparse sector worker context was not initialized.")
    (
        projected_hamiltonian,
        n_particles,
        cutoff,
        sparse_threshold,
        operator_data,
    ) = _PROCESS_SOLVE_CONTEXT
    sector, basis, n_eigs_sector, use_sparse_sector = task
    _, h_many, eigvals, eigvecs = _solve_sector_from_precomputed_basis(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=n_particles,
        cutoff=cutoff,
        n_eigs_sector=n_eigs_sector,
        sparse_threshold=sparse_threshold,
        use_sparse_sector=use_sparse_sector,
        basis=basis,
        operator_data=operator_data,
    )
    return sector, h_many, eigvals, eigvecs


def solve_projected_sector(
    projected_hamiltonian: ProjectedFermionHamiltonian,
    n_particles: int,
    momentum_sector: tuple[int, int] | None = None,
    cutoff: float = 1e-12,
    n_eigs: int = 1,
    sparse_threshold: int = 2000,
    use_sparse: bool = True,
    basis: ManyBodyBasis | None = None,
    operator_data: ManyBodyOperatorData | None = None,
) -> tuple[ManyBodyBasis, np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray]:
    # Written with Codex 02-19-26.
    lattice = projected_hamiltonian.lattice
    n_orbitals = projected_hamiltonian.one_body.shape[0]
    if basis is None:
        basis = build_fock_basis(
            n_orbitals=n_orbitals,
            n_particles=n_particles,
            orbital_momenta=projected_hamiltonian.orbital_momenta,
            lattice_shape=(lattice.Lx, lattice.Ly),
            momentum_sector=momentum_sector,
        )
    elif basis.n_orbitals != int(n_orbitals) or basis.n_particles != int(n_particles):
        raise ValueError(
            "Provided basis is incompatible with requested "
            f"(n_orbitals={n_orbitals}, n_particles={n_particles})."
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
            operator_data=operator_data,
        )
    else:
        h_many = build_many_body_hamiltonian_dense(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=cutoff,
            operator_data=operator_data,
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
    sparse_threshold: int = 200,
    use_sparse: bool = True,
    n_eigs_sparse: int | None = None,
    parallel_sparse_sectors: bool = True,
    max_sector_workers: int | None = 12,
    sparse_parallel_backend: str = "thread",
) -> Mapping[tuple[int, int], dict[str, object]]:
    # Written with Codex 02-19-26.
    lattice = projected_hamiltonian.lattice
    results: dict[tuple[int, int], dict[str, object]] = {}
    backend = _normalize_sparse_parallel_backend(sparse_parallel_backend)

    # Shared preprocessing is done once up front and reused across all sectors.
    operator_data = prepare_many_body_operator_data(
        one_body=projected_hamiltonian.one_body,
        two_body=projected_hamiltonian.two_body,
        cutoff=cutoff,
        build_numba_compact=True,
    )

    sparse_specs: list[tuple[tuple[int, int], ManyBodyBasis, int, bool]] = []
    dense_specs: list[tuple[tuple[int, int], ManyBodyBasis, int, bool]] = []
    for kx in range(lattice.Lx):
        for ky in range(lattice.Ly):
            sector = (kx, ky)
            basis = build_fock_basis(
                n_orbitals=projected_hamiltonian.one_body.shape[0],
                n_particles=n_particles,
                orbital_momenta=projected_hamiltonian.orbital_momenta,
                lattice_shape=(lattice.Lx, lattice.Ly),
                momentum_sector=sector,
            )
            dim = len(basis.states)
            if dim == 0:
                continue

            use_sparse_sector = bool(use_sparse) and (dim > sparse_threshold)
            if use_sparse_sector and n_eigs_sparse is not None:
                n_eigs_sector = int(n_eigs_sparse)
            else:
                n_eigs_sector = int(n_eigs)

            spec = (sector, basis, int(n_eigs_sector), bool(use_sparse_sector))
            if use_sparse_sector:
                sparse_specs.append(spec)
            else:
                dense_specs.append(spec)

    for sector, basis, n_eigs_sector, use_sparse_sector in dense_specs:
        basis, h_many, eigvals, eigvecs = _solve_sector_from_precomputed_basis(
            projected_hamiltonian=projected_hamiltonian,
            n_particles=n_particles,
            cutoff=cutoff,
            n_eigs_sector=n_eigs_sector,
            sparse_threshold=sparse_threshold,
            use_sparse_sector=use_sparse_sector,
            basis=basis,
            operator_data=operator_data,
        )
        results[sector] = {
            "basis": basis,
            "hamiltonian": h_many,
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "solver": "dense",
        }

    if not sparse_specs:
        return results

    run_sparse_in_parallel = bool(parallel_sparse_sectors) and (len(sparse_specs) > 1)
    if run_sparse_in_parallel:
        n_workers = _sector_worker_count(
            n_tasks=len(sparse_specs),
            max_sector_workers=max_sector_workers,
        )
    else:
        n_workers = 1

    if n_workers <= 1:
        for sector, basis, n_eigs_sector, use_sparse_sector in sparse_specs:
            basis, h_many, eigvals, eigvecs = _solve_sector_from_precomputed_basis(
                projected_hamiltonian=projected_hamiltonian,
                n_particles=n_particles,
                cutoff=cutoff,
                n_eigs_sector=n_eigs_sector,
                sparse_threshold=sparse_threshold,
                use_sparse_sector=use_sparse_sector,
                basis=basis,
                operator_data=operator_data,
            )
            results[sector] = {
                "basis": basis,
                "hamiltonian": h_many,
                "eigenvalues": eigvals,
                "eigenvectors": eigvecs,
                "solver": "sparse",
            }
        return results

    if backend == "thread":
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(
                    _solve_sector_from_precomputed_basis,
                    projected_hamiltonian,
                    n_particles,
                    cutoff,
                    n_eigs_sector,
                    sparse_threshold,
                    use_sparse_sector,
                    basis,
                    operator_data,
                ): sector
                for sector, basis, n_eigs_sector, use_sparse_sector in sparse_specs
            }
            for future, sector in future_map.items():
                basis, h_many, eigvals, eigvecs = future.result()
                results[sector] = {
                    "basis": basis,
                    "hamiltonian": h_many,
                    "eigenvalues": eigvals,
                    "eigenvectors": eigvecs,
                    "solver": "sparse",
                }
        return results

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_sparse_sector_process_context,
        initargs=(
            projected_hamiltonian,
            n_particles,
            cutoff,
            sparse_threshold,
            operator_data,
        ),
    ) as pool:
        future_map = {
            pool.submit(_solve_sparse_sector_process_task, spec): spec[0]
            for spec in sparse_specs
        }
        for future, sector in future_map.items():
            _, h_many, eigvals, eigvecs = future.result()
            basis = next(spec_basis for spec_sector, spec_basis, _, _ in sparse_specs if spec_sector == sector)
            results[sector] = {
                "basis": basis,
                "hamiltonian": h_many,
                "eigenvalues": eigvals,
                "eigenvectors": eigvecs,
                "solver": "sparse",
            }

    return results
