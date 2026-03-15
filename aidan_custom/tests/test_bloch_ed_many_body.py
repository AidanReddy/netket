from __future__ import annotations

import numpy as np

from aidan_custom.bloch_ed.many_body import (
    _NUMBA_AVAILABLE,
    _ensure_many_body_operator_data,
    build_all_momentum_sector_bases,
    build_fock_basis,
    build_many_body_hamiltonian_dense,
    build_many_body_hamiltonian_sparse,
    prepare_many_body_operator_data,
)
from aidan_custom.bloch_ed.types import LatticeEmbedding2D, ProjectedFermionHamiltonian
from aidan_custom.bloch_ed.workflow import solve_projected_all_momentum_sectors


def _toy_many_body_inputs():
    # Written with Codex 03-14-26.
    one_body = np.asarray(
        [
            [0.6, 0.1 - 0.2j, 0.0, 0.05],
            [0.1 + 0.2j, -0.3, -0.15j, 0.0],
            [0.0, 0.15j, 0.4, -0.12 + 0.03j],
            [0.05, 0.0, -0.12 - 0.03j, -0.1],
        ],
        dtype=np.complex128,
    )
    two_body = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    two_body[0, 1, 0, 1] = 0.7
    two_body[1, 0, 1, 0] = 0.7
    two_body[2, 3, 2, 3] = -0.2
    two_body[3, 2, 3, 2] = -0.2
    two_body[0, 2, 1, 3] = 0.05 + 0.04j
    two_body[3, 1, 2, 0] = 0.05 - 0.04j
    basis = build_fock_basis(n_orbitals=4, n_particles=2)
    return one_body, two_body, basis


def test_dense_builder_upgrades_noncompact_operator_data():
    # Written with Codex 03-14-26.
    one_body, two_body, basis = _toy_many_body_inputs()
    operator_data_plain = prepare_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        build_numba_compact=False,
    )
    operator_data_compact = prepare_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        build_numba_compact=True,
    )

    h_plain = build_many_body_hamiltonian_dense(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        operator_data=operator_data_plain,
    )
    h_compact = build_many_body_hamiltonian_dense(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        operator_data=operator_data_compact,
    )

    assert np.allclose(h_plain, h_compact)


def test_sparse_builder_upgrades_noncompact_operator_data():
    # Written with Codex 03-14-26.
    one_body, two_body, basis = _toy_many_body_inputs()
    operator_data_plain = prepare_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        build_numba_compact=False,
    )
    operator_data_compact = prepare_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        build_numba_compact=True,
    )

    h_plain = build_many_body_hamiltonian_sparse(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        operator_data=operator_data_plain,
    )
    h_compact = build_many_body_hamiltonian_sparse(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        operator_data=operator_data_compact,
    )

    assert np.allclose(h_plain.toarray(), h_compact.toarray())


def test_operator_data_upgrade_populates_compact_cache_when_available():
    # Written with Codex 03-14-26.
    one_body, two_body, _ = _toy_many_body_inputs()
    operator_data_plain = prepare_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        build_numba_compact=False,
    )

    upgraded = _ensure_many_body_operator_data(
        one_body=one_body,
        two_body=two_body,
        cutoff=1e-12,
        operator_data=operator_data_plain,
        prefer_numba_compact=True,
    )

    assert operator_data_plain.numba_compact is None
    if _NUMBA_AVAILABLE:
        assert upgraded.numba_compact is not None
    else:
        assert upgraded.numba_compact is None


def test_build_all_momentum_sector_bases_matches_per_sector_builder():
    # Written with Codex 03-14-26.
    orbital_momenta = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.int64,
    )
    all_bases = build_all_momentum_sector_bases(
        n_orbitals=4,
        n_particles=2,
        orbital_momenta=orbital_momenta,
        lattice_shape=(2, 2),
    )

    for sector, basis_all in all_bases.items():
        basis_single = build_fock_basis(
            n_orbitals=4,
            n_particles=2,
            orbital_momenta=orbital_momenta,
            lattice_shape=(2, 2),
            momentum_sector=sector,
        )
        assert np.array_equal(basis_all.states, basis_single.states)
        assert basis_all.state_index is None
        assert basis_single.state_index is not None


def test_dense_builder_matches_with_and_without_state_index_dict():
    # Written with Codex 03-14-26.
    one_body, two_body, _ = _toy_many_body_inputs()
    basis_with_index = build_fock_basis(
        n_orbitals=4,
        n_particles=2,
        build_state_index=True,
    )
    basis_without_index = build_fock_basis(
        n_orbitals=4,
        n_particles=2,
        build_state_index=False,
    )

    h_with_index = build_many_body_hamiltonian_dense(
        one_body=one_body,
        two_body=two_body,
        basis=basis_with_index,
    )
    h_without_index = build_many_body_hamiltonian_dense(
        one_body=one_body,
        two_body=two_body,
        basis=basis_without_index,
    )

    assert np.allclose(h_with_index, h_without_index)


def test_all_momentum_sector_solver_matches_serial_and_parallel_threaded():
    # Written with Codex 03-14-26.
    lattice = LatticeEmbedding2D(
        Lx=2,
        Ly=2,
        n_orbitals_per_cell=1,
        site_index=np.zeros((2, 2, 1), dtype=np.int64),
        site_to_cell=np.zeros((4, 2), dtype=np.int64),
        cell_to_bravais=np.zeros((4, 2), dtype=np.int64),
        supercell_matrix=np.eye(2, dtype=np.int64),
        kpoint_coefficients=np.zeros((4, 2), dtype=np.float64),
    )
    one_body = np.asarray(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.1, -0.1, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.07j],
            [0.0, 0.0, -0.07j, -0.4],
        ],
        dtype=np.complex128,
    )
    two_body = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    two_body[0, 1, 0, 1] = 0.25
    two_body[1, 0, 1, 0] = 0.25
    two_body[2, 3, 2, 3] = -0.15
    two_body[3, 2, 3, 2] = -0.15
    projected = ProjectedFermionHamiltonian(
        lattice=lattice,
        orbital_momenta=np.asarray(
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            dtype=np.int64,
        ),
        orbital_bands=np.zeros(4, dtype=np.int64),
        one_body=one_body,
        two_body=two_body,
    )

    sequential = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=2,
        n_eigs=1,
        use_sparse=True,
        sparse_threshold=1,
        parallel_sparse_sectors=False,
    )
    threaded = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected,
        n_particles=2,
        n_eigs=1,
        use_sparse=True,
        sparse_threshold=1,
        parallel_sparse_sectors=True,
        max_sector_workers=2,
        sparse_parallel_backend="thread",
    )

    assert set(sequential) == set(threaded)
    for sector in sequential:
        assert np.allclose(
            sequential[sector]["eigenvalues"],
            threaded[sector]["eigenvalues"],
        )
