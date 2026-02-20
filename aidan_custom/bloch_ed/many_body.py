from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .interactions import extract_nonzero_quartic_terms
from .types import ManyBodyBasis, QuarticTermList


def iter_occupied_orbitals(state: int):
    # Written with Codex 02-19-26.
    bits = int(state)
    while bits:
        lsb = bits & -bits
        yield int(lsb.bit_length() - 1)
        bits ^= lsb


def _parity_sign_before_orbital(state: int, orbital: int) -> int:
    # Written with Codex 02-19-26.
    mask = (1 << int(orbital)) - 1
    parity = (int(state) & mask).bit_count() & 1
    return -1 if parity else 1


def apply_annihilation(state: int, orbital: int) -> tuple[int, int]:
    # Written with Codex 02-19-26.
    bit = 1 << int(orbital)
    if (int(state) & bit) == 0:
        return int(state), 0
    sign = _parity_sign_before_orbital(int(state), int(orbital))
    return int(state) ^ bit, sign


def apply_creation(state: int, orbital: int) -> tuple[int, int]:
    # Written with Codex 02-19-26.
    bit = 1 << int(orbital)
    if (int(state) & bit) != 0:
        return int(state), 0
    sign = _parity_sign_before_orbital(int(state), int(orbital))
    return int(state) | bit, sign


def build_fock_basis(
    n_orbitals: int,
    n_particles: int,
    orbital_momenta: np.ndarray | None = None,
    lattice_shape: tuple[int, int] | None = None,
    momentum_sector: tuple[int, int] | None = None,
) -> ManyBodyBasis:
    # Written with Codex 02-19-26.
    if n_orbitals < 0:
        raise ValueError("n_orbitals must be non-negative.")
    if not (0 <= n_particles <= n_orbitals):
        raise ValueError("n_particles must satisfy 0 <= n_particles <= n_orbitals.")

    use_momentum = momentum_sector is not None
    if use_momentum:
        if orbital_momenta is None or lattice_shape is None:
            raise ValueError(
                "Need orbital_momenta and lattice_shape when momentum_sector is set."
            )
        if orbital_momenta.shape != (n_orbitals, 2):
            raise ValueError(
                "orbital_momenta must have shape "
                f"({n_orbitals}, 2), got {orbital_momenta.shape}."
            )
        Lx, Ly = lattice_shape
        target_kx = int(momentum_sector[0]) % int(Lx)
        target_ky = int(momentum_sector[1]) % int(Ly)
        orbital_kx = np.asarray(orbital_momenta[:, 0], dtype=np.int64)
        orbital_ky = np.asarray(orbital_momenta[:, 1], dtype=np.int64)
    else:
        Lx = Ly = 1
        target_kx = target_ky = 0
        orbital_kx = orbital_ky = np.asarray([], dtype=np.int64)

    states: list[int] = []
    for occ in combinations(range(n_orbitals), n_particles):
        if use_momentum:
            kx = int(np.sum(orbital_kx[list(occ)])) % int(Lx)
            ky = int(np.sum(orbital_ky[list(occ)])) % int(Ly)
            if (kx, ky) != (target_kx, target_ky):
                continue

        state = 0
        for orb in occ:
            state |= 1 << int(orb)
        states.append(int(state))

    state_index = {state: idx for idx, state in enumerate(states)}
    sector = (target_kx, target_ky) if use_momentum else None
    return ManyBodyBasis(
        n_orbitals=int(n_orbitals),
        n_particles=int(n_particles),
        states=states,
        state_index=state_index,
        momentum_sector=sector,
    )


def _group_one_body_terms(
    one_body: np.ndarray,
    cutoff: float,
) -> dict[int, list[tuple[int, complex]]]:
    # Written with Codex 02-19-26.
    grouped: dict[int, list[tuple[int, complex]]] = defaultdict(list)
    nz = np.argwhere(np.abs(one_body) > cutoff)
    for a, b in nz:
        grouped[int(b)].append((int(a), np.complex128(one_body[a, b])))
    return grouped


def _group_quartic_terms(
    terms: QuarticTermList,
) -> dict[tuple[int, int], list[tuple[int, int, complex]]]:
    # Written with Codex 02-19-26.
    grouped: dict[tuple[int, int], list[tuple[int, int, complex]]] = defaultdict(list)
    n_terms = int(terms.coefficients.size)
    for idx in range(n_terms):
        c = int(terms.annihilate_1[idx])
        d = int(terms.annihilate_2[idx])
        a = int(terms.create_1[idx])
        b = int(terms.create_2[idx])
        coeff = np.complex128(terms.coefficients[idx])
        grouped[(c, d)].append((a, b, coeff))
    return grouped


def _accumulate_many_body_entries(
    one_body: np.ndarray,
    two_body: np.ndarray,
    basis: ManyBodyBasis,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-19-26.
    n_orb = basis.n_orbitals
    if one_body.shape != (n_orb, n_orb):
        raise ValueError(
            f"one_body must have shape ({n_orb}, {n_orb}), got {one_body.shape}."
        )
    if two_body.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError(
            "two_body must have shape "
            f"({n_orb}, {n_orb}, {n_orb}, {n_orb}), got {two_body.shape}."
        )

    one_grouped = _group_one_body_terms(one_body, cutoff=cutoff)
    quartic_terms = extract_nonzero_quartic_terms(two_body, cutoff=cutoff)
    two_grouped = _group_quartic_terms(quartic_terms)

    entries: dict[tuple[int, int], np.complex128] = defaultdict(np.complex128)
    for col, state in enumerate(basis.states):
        occ = tuple(iter_occupied_orbitals(state))

        for b in occ:
            terms_b = one_grouped.get(int(b))
            if terms_b is None:
                continue
            state_b, sign_b = apply_annihilation(state, int(b))
            for a, coeff in terms_b:
                state_ab, sign_a = apply_creation(state_b, a)
                if sign_a == 0:
                    continue
                row = basis.state_index.get(state_ab)
                if row is not None:
                    entries[(int(row), int(col))] += coeff * sign_b * sign_a

        for c in occ:
            state_c, sign_c = apply_annihilation(state, int(c))
            occ_after_c = tuple(iter_occupied_orbitals(state_c))
            for d in occ_after_c:
                terms_cd = two_grouped.get((int(c), int(d)))
                if terms_cd is None:
                    continue
                state_cd, sign_d = apply_annihilation(state_c, int(d))
                sign_cd = sign_c * sign_d
                for a, b, coeff in terms_cd:
                    state_cdb, sign_b = apply_creation(state_cd, b)
                    if sign_b == 0:
                        continue
                    state_final, sign_a = apply_creation(state_cdb, a)
                    if sign_a == 0:
                        continue
                    row = basis.state_index.get(state_final)
                    if row is not None:
                        entries[(int(row), int(col))] += coeff * sign_cd * sign_b * sign_a

    if len(entries) == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.complex128),
        )

    rows = np.empty(len(entries), dtype=np.int64)
    cols = np.empty(len(entries), dtype=np.int64)
    data = np.empty(len(entries), dtype=np.complex128)
    for idx, ((row, col), value) in enumerate(entries.items()):
        rows[idx] = int(row)
        cols[idx] = int(col)
        data[idx] = np.complex128(value)

    return rows, cols, data


def build_many_body_hamiltonian_sparse(
    one_body: np.ndarray,
    two_body: np.ndarray,
    basis: ManyBodyBasis,
    cutoff: float = 1e-12,
    hermitize: bool = True,
) -> sp.csr_matrix:
    # Written with Codex 02-19-26.
    rows, cols, data = _accumulate_many_body_entries(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        cutoff=cutoff,
    )
    dim = len(basis.states)
    h_sparse = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    if hermitize:
        h_sparse = 0.5 * (h_sparse + h_sparse.getH())
    return h_sparse


def build_many_body_hamiltonian_dense(
    one_body: np.ndarray,
    two_body: np.ndarray,
    basis: ManyBodyBasis,
    cutoff: float = 1e-12,
    hermitize: bool = True,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    rows, cols, data = _accumulate_many_body_entries(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        cutoff=cutoff,
    )
    dim = len(basis.states)
    h_dense = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(data.size):
        h_dense[rows[idx], cols[idx]] += data[idx]
    if hermitize:
        h_dense = 0.5 * (h_dense + np.conjugate(h_dense.T))
    return h_dense


def diagonalize_many_body_hamiltonian(
    hamiltonian: np.ndarray | sp.spmatrix,
    n_eigs: int = 1,
    sparse_threshold: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-19-26.
    if n_eigs <= 0:
        raise ValueError("n_eigs must be positive.")

    if sp.issparse(hamiltonian):
        dim = int(hamiltonian.shape[0])
    else:
        dim = int(np.asarray(hamiltonian).shape[0])

    if dim == 0:
        raise ValueError("Hamiltonian dimension is zero.")
    if n_eigs > dim:
        raise ValueError(f"n_eigs={n_eigs} exceeds matrix dimension dim={dim}.")

    if (
        sp.issparse(hamiltonian)
        and dim > sparse_threshold
        and n_eigs < (dim - 1)
    ):
        eigvals, eigvecs = eigsh(
            hamiltonian,
            k=int(n_eigs),
            which="SA",
            return_eigenvectors=True,
            tol=1e-10,
        )
        order = np.argsort(np.real(eigvals))
        return np.real(eigvals[order]), eigvecs[:, order]

    dense = (
        hamiltonian.toarray()
        if sp.issparse(hamiltonian)
        else np.asarray(hamiltonian, dtype=np.complex128)
    )
    eigvals, eigvecs = np.linalg.eigh(dense)
    return np.real(eigvals[:n_eigs]), eigvecs[:, :n_eigs]
