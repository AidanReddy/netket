from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .interactions import extract_nonzero_quartic_terms
from .types import ManyBodyBasis, QuarticTermList

try:
    import numba as nb

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    nb = None
    _NUMBA_AVAILABLE = False

_MOMENTUM_SECTOR_STATES_CACHE: dict[
    tuple[int, int, int, int, bytes],
    dict[tuple[int, int], np.ndarray],
] = {}


@dataclass(frozen=True)
class ManyBodyOperatorDataNumbaCompact:
    one_offsets: np.ndarray
    one_targets: np.ndarray
    one_coeff_re: np.ndarray
    one_coeff_im: np.ndarray
    d_offsets: np.ndarray
    d_values: np.ndarray
    pair_offsets: np.ndarray
    two_create_a: np.ndarray
    two_create_b: np.ndarray
    two_coeff_re: np.ndarray
    two_coeff_im: np.ndarray


@dataclass(frozen=True)
class ManyBodyOperatorData:
    one_grouped: dict[int, list[tuple[int, complex]]]
    two_grouped: dict[tuple[int, int], list[tuple[int, int, complex]]]
    two_targets_by_c: dict[int, tuple[int, ...]]
    numba_compact: ManyBodyOperatorDataNumbaCompact | None = None



def _build_occupied_matrix(states: np.ndarray, n_particles: int) -> np.ndarray:
    # Written with Codex 02-21-26.
    n_states = int(states.size)
    n_part = int(n_particles)
    occupied = np.empty((n_states, n_part), dtype=np.int64)
    for idx in range(n_states):
        bits = int(states[idx])
        out = 0
        while bits:
            lsb = bits & -bits
            occupied[idx, out] = int(lsb.bit_length() - 1)
            bits ^= lsb
            out += 1
        if out != n_part:
            raise ValueError(
                "State occupation does not match basis.n_particles: "
                f"expected {n_part}, got {out}."
            )
    return occupied


def _build_one_body_compact_arrays(
    one_grouped: dict[int, list[tuple[int, complex]]],
    n_orbitals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    n_orb = int(n_orbitals)
    counts = np.zeros(n_orb, dtype=np.int64)
    for b, terms in one_grouped.items():
        counts[int(b)] = int(len(terms))

    offsets = np.empty(n_orb + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])

    targets = np.empty(total, dtype=np.int64)
    coeff_re = np.empty(total, dtype=np.float64)
    coeff_im = np.empty(total, dtype=np.float64)
    write_ptr = offsets[:-1].copy()
    for b, terms in one_grouped.items():
        ptr = int(write_ptr[int(b)])
        for a, coeff in terms:
            value = complex(coeff)
            targets[ptr] = int(a)
            coeff_re[ptr] = float(value.real)
            coeff_im[ptr] = float(value.imag)
            ptr += 1
        write_ptr[int(b)] = ptr

    return offsets, targets, coeff_re, coeff_im


def _build_two_body_compact_arrays(
    two_grouped: dict[tuple[int, int], list[tuple[int, int, complex]]],
    two_targets_by_c: dict[int, tuple[int, ...]],
    n_orbitals: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # Written with Codex 02-21-26.
    n_orb = int(n_orbitals)
    n_pairs = n_orb * n_orb
    pair_counts = np.zeros(n_pairs, dtype=np.int64)
    for (c, d), terms in two_grouped.items():
        pair_counts[int(c) * n_orb + int(d)] = int(len(terms))

    pair_offsets = np.empty(n_pairs + 1, dtype=np.int64)
    pair_offsets[0] = 0
    np.cumsum(pair_counts, out=pair_offsets[1:])
    total_terms = int(pair_offsets[-1])

    create_a = np.empty(total_terms, dtype=np.int64)
    create_b = np.empty(total_terms, dtype=np.int64)
    coeff_re = np.empty(total_terms, dtype=np.float64)
    coeff_im = np.empty(total_terms, dtype=np.float64)
    pair_write = pair_offsets[:-1].copy()

    for (c, d), terms in two_grouped.items():
        pair_idx = int(c) * n_orb + int(d)
        ptr = int(pair_write[pair_idx])
        for a, b, coeff in terms:
            value = complex(coeff)
            create_a[ptr] = int(a)
            create_b[ptr] = int(b)
            coeff_re[ptr] = float(value.real)
            coeff_im[ptr] = float(value.imag)
            ptr += 1
        pair_write[pair_idx] = ptr

    d_counts = np.zeros(n_orb, dtype=np.int64)
    for c, d_targets in two_targets_by_c.items():
        d_counts[int(c)] = int(len(d_targets))
    d_offsets = np.empty(n_orb + 1, dtype=np.int64)
    d_offsets[0] = 0
    np.cumsum(d_counts, out=d_offsets[1:])
    n_d_values = int(d_offsets[-1])
    d_values = np.empty(n_d_values, dtype=np.int64)
    d_write = d_offsets[:-1].copy()
    for c, d_targets in two_targets_by_c.items():
        ptr = int(d_write[int(c)])
        for d in d_targets:
            d_values[ptr] = int(d)
            ptr += 1
        d_write[int(c)] = ptr

    return d_offsets, d_values, pair_offsets, create_a, create_b, coeff_re, coeff_im


if _NUMBA_AVAILABLE:

    @nb.njit(cache=True, inline="always")
    def _numba_popcount(bits: int) -> int:
        # Written with Codex 02-21-26.
        count = 0
        value = bits
        while value != 0:
            value &= value - 1
            count += 1
        return count

    @nb.njit(cache=True, inline="always")
    def _numba_bsearch(sorted_states: np.ndarray, target: np.int64) -> np.int64:
        # Binary search on sorted int64 state array; returns index or -1 if absent.
        lo = np.int64(0)
        hi = np.int64(sorted_states.shape[0] - 1)
        while lo <= hi:
            mid = (lo + hi) >> np.int64(1)
            val = sorted_states[mid]
            if val == target:
                return mid
            elif val < target:
                lo = mid + np.int64(1)
            else:
                hi = mid - np.int64(1)
        return np.int64(-1)

    @nb.njit(cache=True)
    def _assemble_dense_hamiltonian_numba(
        states: np.ndarray,
        occupied: np.ndarray,
        orbital_bits: np.ndarray,
        lower_masks: np.ndarray,
        one_offsets: np.ndarray,
        one_targets: np.ndarray,
        one_coeff_re: np.ndarray,
        one_coeff_im: np.ndarray,
        d_offsets: np.ndarray,
        d_values: np.ndarray,
        pair_offsets: np.ndarray,
        two_create_a: np.ndarray,
        two_create_b: np.ndarray,
        two_coeff_re: np.ndarray,
        two_coeff_im: np.ndarray,
        n_orbitals: int,
    ) -> np.ndarray:
        # Written with Codex 02-21-26.
        dim = int(states.shape[0])
        n_particles = int(occupied.shape[1])
        h_dense = np.zeros((dim, dim), dtype=np.complex128)
        ann_states = np.empty(n_particles, dtype=np.int64)
        ann_signs = np.empty(n_particles, dtype=np.int64)

        for col in range(dim):
            state = int(states[col])
            for occ_idx in range(n_particles):
                orb = int(occupied[col, occ_idx])
                bit = int(orbital_bits[orb])
                ann_states[occ_idx] = state ^ bit
                parity = _numba_popcount(state & int(lower_masks[orb])) & 1
                ann_signs[occ_idx] = -1 if parity == 1 else 1

            for occ_idx in range(n_particles):
                b = int(occupied[col, occ_idx])
                state_b = int(ann_states[occ_idx])
                sign_b = int(ann_signs[occ_idx])
                start = int(one_offsets[b])
                end = int(one_offsets[b + 1])
                for term_idx in range(start, end):
                    a = int(one_targets[term_idx])
                    bit_a = int(orbital_bits[a])
                    if (state_b & bit_a) != 0:
                        continue
                    parity_a = _numba_popcount(state_b & int(lower_masks[a])) & 1
                    sign_a = -1 if parity_a == 1 else 1
                    row_state = state_b | bit_a
                    row = int(_numba_bsearch(states, np.int64(row_state)))
                    if row >= 0:
                        coeff = complex(
                            one_coeff_re[term_idx],
                            one_coeff_im[term_idx],
                        )
                        h_dense[row, col] += coeff * float(sign_b * sign_a)

            for occ_idx in range(n_particles):
                c = int(occupied[col, occ_idx])
                state_c = int(ann_states[occ_idx])
                sign_c = int(ann_signs[occ_idx])
                d_start = int(d_offsets[c])
                d_end = int(d_offsets[c + 1])
                for d_idx in range(d_start, d_end):
                    d = int(d_values[d_idx])
                    bit_d = int(orbital_bits[d])
                    if (state_c & bit_d) == 0:
                        continue
                    parity_d = _numba_popcount(state_c & int(lower_masks[d])) & 1
                    sign_d = -1 if parity_d == 1 else 1
                    state_cd = state_c ^ bit_d
                    sign_cd = sign_c * sign_d

                    pair_idx = c * n_orbitals + d
                    term_start = int(pair_offsets[pair_idx])
                    term_end = int(pair_offsets[pair_idx + 1])
                    for term_idx in range(term_start, term_end):
                        b = int(two_create_b[term_idx])
                        bit_b = int(orbital_bits[b])
                        if (state_cd & bit_b) != 0:
                            continue
                        parity_b = _numba_popcount(state_cd & int(lower_masks[b])) & 1
                        sign_b = -1 if parity_b == 1 else 1
                        state_cdb = state_cd | bit_b

                        a = int(two_create_a[term_idx])
                        bit_a = int(orbital_bits[a])
                        if (state_cdb & bit_a) != 0:
                            continue
                        parity_a = _numba_popcount(state_cdb & int(lower_masks[a])) & 1
                        sign_a = -1 if parity_a == 1 else 1
                        row_state = state_cdb | bit_a
                        row = int(_numba_bsearch(states, np.int64(row_state)))
                        if row >= 0:
                            coeff = complex(
                                two_coeff_re[term_idx],
                                two_coeff_im[term_idx],
                            )
                            h_dense[row, col] += coeff * float(
                                sign_cd * sign_b * sign_a
                            )

        return h_dense

    @nb.njit(cache=True, nogil=True)
    def _count_sparse_entries_numba(
        states: np.ndarray,
        occupied: np.ndarray,
        orbital_bits: np.ndarray,
        one_offsets: np.ndarray,
        one_targets: np.ndarray,
        d_offsets: np.ndarray,
        d_values: np.ndarray,
        pair_offsets: np.ndarray,
        two_create_a: np.ndarray,
        two_create_b: np.ndarray,
        n_orbitals: int,
    ) -> int:
        # Written with Codex 02-21-26.
        # Counts COO triplets produced by the sparse Hamiltonian fill pass.
        # Signs and coefficients are irrelevant for counting; only structural
        # connectivity (which operator applications land in the basis) matters.
        dim = int(states.shape[0])
        n_particles = int(occupied.shape[1])
        count = 0
        ann_states = np.empty(n_particles, dtype=np.int64)

        for col in range(dim):
            state = int(states[col])
            for occ_idx in range(n_particles):
                orb = int(occupied[col, occ_idx])
                ann_states[occ_idx] = state ^ int(orbital_bits[orb])

            for occ_idx in range(n_particles):
                b = int(occupied[col, occ_idx])
                state_b = int(ann_states[occ_idx])
                start = int(one_offsets[b])
                end = int(one_offsets[b + 1])
                for term_idx in range(start, end):
                    a = int(one_targets[term_idx])
                    bit_a = int(orbital_bits[a])
                    if (state_b & bit_a) != 0:
                        continue
                    if _numba_bsearch(states, np.int64(state_b | bit_a)) >= 0:
                        count += 1

            for occ_idx in range(n_particles):
                c = int(occupied[col, occ_idx])
                state_c = int(ann_states[occ_idx])
                d_start = int(d_offsets[c])
                d_end = int(d_offsets[c + 1])
                for d_idx in range(d_start, d_end):
                    d = int(d_values[d_idx])
                    bit_d = int(orbital_bits[d])
                    if (state_c & bit_d) == 0:
                        continue
                    state_cd = state_c ^ bit_d
                    pair_idx = c * n_orbitals + d
                    term_start = int(pair_offsets[pair_idx])
                    term_end = int(pair_offsets[pair_idx + 1])
                    for term_idx in range(term_start, term_end):
                        b = int(two_create_b[term_idx])
                        bit_b = int(orbital_bits[b])
                        if (state_cd & bit_b) != 0:
                            continue
                        state_cdb = state_cd | bit_b
                        a = int(two_create_a[term_idx])
                        bit_a = int(orbital_bits[a])
                        if (state_cdb & bit_a) != 0:
                            continue
                        if _numba_bsearch(states, np.int64(state_cdb | bit_a)) >= 0:
                            count += 1

        return count

    @nb.njit(cache=True, nogil=True)
    def _fill_sparse_entries_numba(
        row_out: np.ndarray,
        col_out: np.ndarray,
        data_re_out: np.ndarray,
        data_im_out: np.ndarray,
        states: np.ndarray,
        occupied: np.ndarray,
        orbital_bits: np.ndarray,
        lower_masks: np.ndarray,
        one_offsets: np.ndarray,
        one_targets: np.ndarray,
        one_coeff_re: np.ndarray,
        one_coeff_im: np.ndarray,
        d_offsets: np.ndarray,
        d_values: np.ndarray,
        pair_offsets: np.ndarray,
        two_create_a: np.ndarray,
        two_create_b: np.ndarray,
        two_coeff_re: np.ndarray,
        two_coeff_im: np.ndarray,
        n_orbitals: int,
    ) -> int:
        # Written with Codex 02-21-26.
        # Fills pre-allocated COO arrays. Duplicates are allowed; scipy sums
        # them automatically on COO -> CSR conversion (e.g. diagonal terms
        # contributed by multiple one-body or two-body paths).
        dim = int(states.shape[0])
        n_particles = int(occupied.shape[1])
        ptr = 0
        ann_states = np.empty(n_particles, dtype=np.int64)
        ann_signs = np.empty(n_particles, dtype=np.int64)

        for col in range(dim):
            state = int(states[col])
            for occ_idx in range(n_particles):
                orb = int(occupied[col, occ_idx])
                bit = int(orbital_bits[orb])
                ann_states[occ_idx] = state ^ bit
                parity = _numba_popcount(state & int(lower_masks[orb])) & 1
                ann_signs[occ_idx] = -1 if parity == 1 else 1

            for occ_idx in range(n_particles):
                b = int(occupied[col, occ_idx])
                state_b = int(ann_states[occ_idx])
                sign_b = int(ann_signs[occ_idx])
                start = int(one_offsets[b])
                end = int(one_offsets[b + 1])
                for term_idx in range(start, end):
                    a = int(one_targets[term_idx])
                    bit_a = int(orbital_bits[a])
                    if (state_b & bit_a) != 0:
                        continue
                    parity_a = _numba_popcount(state_b & int(lower_masks[a])) & 1
                    sign_a = -1 if parity_a == 1 else 1
                    row_state = state_b | bit_a
                    row = int(_numba_bsearch(states, np.int64(row_state)))
                    if row >= 0:
                        total_sign = float(sign_b * sign_a)
                        row_out[ptr] = row
                        col_out[ptr] = col
                        data_re_out[ptr] = one_coeff_re[term_idx] * total_sign
                        data_im_out[ptr] = one_coeff_im[term_idx] * total_sign
                        ptr += 1

            for occ_idx in range(n_particles):
                c = int(occupied[col, occ_idx])
                state_c = int(ann_states[occ_idx])
                sign_c = int(ann_signs[occ_idx])
                d_start = int(d_offsets[c])
                d_end = int(d_offsets[c + 1])
                for d_idx in range(d_start, d_end):
                    d = int(d_values[d_idx])
                    bit_d = int(orbital_bits[d])
                    if (state_c & bit_d) == 0:
                        continue
                    parity_d = _numba_popcount(state_c & int(lower_masks[d])) & 1
                    sign_d = -1 if parity_d == 1 else 1
                    state_cd = state_c ^ bit_d
                    sign_cd = sign_c * sign_d

                    pair_idx = c * n_orbitals + d
                    term_start = int(pair_offsets[pair_idx])
                    term_end = int(pair_offsets[pair_idx + 1])
                    for term_idx in range(term_start, term_end):
                        b = int(two_create_b[term_idx])
                        bit_b = int(orbital_bits[b])
                        if (state_cd & bit_b) != 0:
                            continue
                        parity_b = _numba_popcount(state_cd & int(lower_masks[b])) & 1
                        sign_b = -1 if parity_b == 1 else 1
                        state_cdb = state_cd | bit_b

                        a = int(two_create_a[term_idx])
                        bit_a = int(orbital_bits[a])
                        if (state_cdb & bit_a) != 0:
                            continue
                        parity_a = _numba_popcount(state_cdb & int(lower_masks[a])) & 1
                        sign_a = -1 if parity_a == 1 else 1
                        row_state = state_cdb | bit_a
                        row = int(_numba_bsearch(states, np.int64(row_state)))
                        if row >= 0:
                            total_sign = float(sign_cd * sign_b * sign_a)
                            row_out[ptr] = row
                            col_out[ptr] = col
                            data_re_out[ptr] = two_coeff_re[term_idx] * total_sign
                            data_im_out[ptr] = two_coeff_im[term_idx] * total_sign
                            ptr += 1

        return ptr

else:

    _assemble_dense_hamiltonian_numba = None
    _count_sparse_entries_numba = None
    _fill_sparse_entries_numba = None


def _build_many_body_hamiltonian_dense_numba(
    basis: ManyBodyBasis,
    operator_data: ManyBodyOperatorData,
) -> np.ndarray | None:
    # Written with Codex 02-21-26.
    if not _NUMBA_AVAILABLE:
        return None
    if _assemble_dense_hamiltonian_numba is None:
        return None

    n_orb = int(basis.n_orbitals)
    states = np.asarray(basis.states, dtype=np.int64)
    dim = int(states.size)
    if dim == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    compact = operator_data.numba_compact
    if compact is None:
        return None

    occupied = _build_occupied_matrix(states=states, n_particles=basis.n_particles)
    orbital_bits = np.asarray([1 << orb for orb in range(n_orb)], dtype=np.int64)
    lower_masks = orbital_bits - np.int64(1)
    return _assemble_dense_hamiltonian_numba(
        states=states,
        occupied=occupied,
        orbital_bits=orbital_bits,
        lower_masks=lower_masks,
        one_offsets=compact.one_offsets,
        one_targets=compact.one_targets,
        one_coeff_re=compact.one_coeff_re,
        one_coeff_im=compact.one_coeff_im,
        d_offsets=compact.d_offsets,
        d_values=compact.d_values,
        pair_offsets=compact.pair_offsets,
        two_create_a=compact.two_create_a,
        two_create_b=compact.two_create_b,
        two_coeff_re=compact.two_coeff_re,
        two_coeff_im=compact.two_coeff_im,
        n_orbitals=n_orb,
    )


def _build_many_body_hamiltonian_sparse_numba(
    basis: ManyBodyBasis,
    operator_data: ManyBodyOperatorData,
) -> sp.csr_matrix | None:
    # Written with Codex 02-21-26.
    if not _NUMBA_AVAILABLE:
        return None
    if _count_sparse_entries_numba is None or _fill_sparse_entries_numba is None:
        return None

    compact = operator_data.numba_compact
    if compact is None:
        return None

    n_orb = int(basis.n_orbitals)
    states = np.asarray(basis.states, dtype=np.int64)
    dim = int(states.size)
    if dim == 0:
        return sp.csr_matrix((0, 0), dtype=np.complex128)

    occupied = _build_occupied_matrix(states=states, n_particles=basis.n_particles)
    orbital_bits = np.asarray([1 << orb for orb in range(n_orb)], dtype=np.int64)
    lower_masks = orbital_bits - np.int64(1)

    nnz = int(
        _count_sparse_entries_numba(
            states,
            occupied,
            orbital_bits,
            compact.one_offsets,
            compact.one_targets,
            compact.d_offsets,
            compact.d_values,
            compact.pair_offsets,
            compact.two_create_a,
            compact.two_create_b,
            n_orb,
        )
    )

    row_out = np.empty(nnz, dtype=np.int64)
    col_out = np.empty(nnz, dtype=np.int64)
    data_re = np.empty(nnz, dtype=np.float64)
    data_im = np.empty(nnz, dtype=np.float64)

    _fill_sparse_entries_numba(
        row_out,
        col_out,
        data_re,
        data_im,
        states,
        occupied,
        orbital_bits,
        lower_masks,
        compact.one_offsets,
        compact.one_targets,
        compact.one_coeff_re,
        compact.one_coeff_im,
        compact.d_offsets,
        compact.d_values,
        compact.pair_offsets,
        compact.two_create_a,
        compact.two_create_b,
        compact.two_coeff_re,
        compact.two_coeff_im,
        n_orb,
    )

    data = data_re + 1j * data_im
    return sp.coo_matrix((data, (row_out, col_out)), shape=(dim, dim)).tocsr()


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


def _momentum_sector_cache_key(
    n_orbitals: int,
    n_particles: int,
    orbital_momenta: np.ndarray,
    lattice_shape: tuple[int, int],
) -> tuple[int, int, int, int, bytes]:
    # Written with Codex 02-21-26.
    Lx, Ly = lattice_shape
    momenta = np.asarray(orbital_momenta, dtype=np.int64)
    return (
        int(n_orbitals),
        int(n_particles),
        int(Lx),
        int(Ly),
        momenta.tobytes(order="C"),
    )


def _build_momentum_sector_state_lists(
    n_orbitals: int,
    n_particles: int,
    orbital_momenta: np.ndarray,
    lattice_shape: tuple[int, int],
) -> dict[tuple[int, int], np.ndarray]:
    # Written with Codex 02-21-26.
    Lx, Ly = int(lattice_shape[0]), int(lattice_shape[1])
    orbital_kx = np.asarray(orbital_momenta[:, 0], dtype=np.int64)
    orbital_ky = np.asarray(orbital_momenta[:, 1], dtype=np.int64)

    sector_lists: dict[tuple[int, int], list[int]] = defaultdict(list)
    for occ in combinations(range(int(n_orbitals)), int(n_particles)):
        kx = int(np.sum(orbital_kx[list(occ)])) % Lx
        ky = int(np.sum(orbital_ky[list(occ)])) % Ly
        state = 0
        for orb in occ:
            state |= 1 << int(orb)
        sector_lists[(kx, ky)].append(int(state))

    sector_states: dict[tuple[int, int], np.ndarray] = {}
    for kx in range(Lx):
        for ky in range(Ly):
            sector = (int(kx), int(ky))
            values = sector_lists.get(sector, [])
            sector_states[sector] = np.asarray(values, dtype=np.int64)
    return sector_states


def _get_or_build_momentum_sector_state_lists(
    n_orbitals: int,
    n_particles: int,
    orbital_momenta: np.ndarray,
    lattice_shape: tuple[int, int],
) -> dict[tuple[int, int], np.ndarray]:
    # Written with Codex 02-21-26.
    cache_key = _momentum_sector_cache_key(
        n_orbitals=n_orbitals,
        n_particles=n_particles,
        orbital_momenta=orbital_momenta,
        lattice_shape=lattice_shape,
    )
    cached = _MOMENTUM_SECTOR_STATES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    built = _build_momentum_sector_state_lists(
        n_orbitals=n_orbitals,
        n_particles=n_particles,
        orbital_momenta=orbital_momenta,
        lattice_shape=lattice_shape,
    )
    _MOMENTUM_SECTOR_STATES_CACHE[cache_key] = built
    return built


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
    else:
        Lx = Ly = 1
        target_kx = target_ky = 0

    if use_momentum:
        sector_states = _get_or_build_momentum_sector_state_lists(
            n_orbitals=n_orbitals,
            n_particles=n_particles,
            orbital_momenta=np.asarray(orbital_momenta, dtype=np.int64),
            lattice_shape=(int(Lx), int(Ly)),
        )
        states = sector_states.get(
            (int(target_kx), int(target_ky)),
            np.asarray([], dtype=np.int64),
        )
    else:
        states_list: list[int] = []
        for occ in combinations(range(n_orbitals), n_particles):
            state = 0
            for orb in occ:
                state |= 1 << int(orb)
            states_list.append(int(state))
        states = states_list

    # Sort by state value so Numba kernels can use binary search for O(log dim) lookup.
    states = np.sort(np.asarray(states, dtype=np.int64))
    state_index = {int(state): idx for idx, state in enumerate(states)}
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


def _group_quartic_annihilation_targets(
    grouped_terms: dict[tuple[int, int], list[tuple[int, int, complex]]],
) -> dict[int, tuple[int, ...]]:
    # Written with Codex 02-21-26.
    targets: dict[int, set[int]] = defaultdict(set)
    for c, d in grouped_terms:
        targets[int(c)].add(int(d))
    return {
        int(c): tuple(sorted(int(d) for d in d_set))
        for c, d_set in targets.items()
    }


def prepare_many_body_operator_data(
    one_body: np.ndarray,
    two_body: np.ndarray,
    cutoff: float = 1e-12,
    build_numba_compact: bool = False,
) -> ManyBodyOperatorData:
    # Written with Codex 02-21-26.
    one_body_arr = np.asarray(one_body, dtype=np.complex128)
    two_body_arr = np.asarray(two_body, dtype=np.complex128)

    if one_body_arr.ndim != 2 or one_body_arr.shape[0] != one_body_arr.shape[1]:
        raise ValueError("one_body must have shape (M, M).")
    n_orb = int(one_body_arr.shape[0])
    if two_body_arr.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError(
            "two_body must have shape "
            f"({n_orb}, {n_orb}, {n_orb}, {n_orb}), got {two_body_arr.shape}."
        )

    one_grouped = _group_one_body_terms(one_body_arr, cutoff=float(cutoff))
    quartic_terms = extract_nonzero_quartic_terms(two_body_arr, cutoff=float(cutoff))
    two_grouped = _group_quartic_terms(quartic_terms)
    two_targets_by_c = _group_quartic_annihilation_targets(two_grouped)
    numba_compact = None
    if _NUMBA_AVAILABLE and bool(build_numba_compact):
        one_offsets, one_targets, one_coeff_re, one_coeff_im = _build_one_body_compact_arrays(
            one_grouped=one_grouped,
            n_orbitals=n_orb,
        )
        (
            d_offsets,
            d_values,
            pair_offsets,
            two_create_a,
            two_create_b,
            two_coeff_re,
            two_coeff_im,
        ) = _build_two_body_compact_arrays(
            two_grouped=two_grouped,
            two_targets_by_c=two_targets_by_c,
            n_orbitals=n_orb,
        )
        numba_compact = ManyBodyOperatorDataNumbaCompact(
            one_offsets=one_offsets,
            one_targets=one_targets,
            one_coeff_re=one_coeff_re,
            one_coeff_im=one_coeff_im,
            d_offsets=d_offsets,
            d_values=d_values,
            pair_offsets=pair_offsets,
            two_create_a=two_create_a,
            two_create_b=two_create_b,
            two_coeff_re=two_coeff_re,
            two_coeff_im=two_coeff_im,
        )
    return ManyBodyOperatorData(
        one_grouped=one_grouped,
        two_grouped=two_grouped,
        two_targets_by_c=two_targets_by_c,
        numba_compact=numba_compact,
    )


def _accumulate_many_body_entries(
    one_body: np.ndarray,
    two_body: np.ndarray,
    basis: ManyBodyBasis,
    cutoff: float,
    operator_data: ManyBodyOperatorData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
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

    if operator_data is None:
        operator_data = prepare_many_body_operator_data(
            one_body=one_body,
            two_body=two_body,
            cutoff=cutoff,
        )
    one_grouped = operator_data.one_grouped
    two_grouped = operator_data.two_grouped
    two_targets_by_c = operator_data.two_targets_by_c

    entries: dict[tuple[int, int], np.complex128] = defaultdict(np.complex128)
    state_index_get = basis.state_index.get
    one_grouped_get = one_grouped.get
    two_grouped_get = two_grouped.get
    two_targets_by_c_get = two_targets_by_c.get
    occupied_cache = [tuple(iter_occupied_orbitals(state)) for state in basis.states]
    orbital_bits = tuple(1 << orb for orb in range(n_orb))
    lower_masks = tuple(bit - 1 for bit in orbital_bits)

    for col, state in enumerate(basis.states):
        state_int = int(state)
        occ = occupied_cache[col]

        ann_states: list[int] = [0] * len(occ)
        ann_signs: list[int] = [0] * len(occ)
        for occ_idx, orb in enumerate(occ):
            bit = orbital_bits[orb]
            ann_states[occ_idx] = state_int ^ bit
            parity = (state_int & lower_masks[orb]).bit_count() & 1
            ann_signs[occ_idx] = -1 if parity else 1

        for occ_idx, b_int in enumerate(occ):
            terms_b = one_grouped_get(b_int)
            if terms_b is None:
                continue
            state_b = ann_states[occ_idx]
            sign_b = ann_signs[occ_idx]
            for a, coeff in terms_b:
                bit_a = orbital_bits[a]
                if (state_b & bit_a) != 0:
                    continue
                parity_a = (state_b & lower_masks[a]).bit_count() & 1
                sign_a = -1 if parity_a else 1
                state_ab = state_b | bit_a
                row = state_index_get(state_ab)
                if row is not None:
                    entries[(row, col)] += coeff * sign_b * sign_a

        for occ_idx, c_int in enumerate(occ):
            d_candidates = two_targets_by_c_get(c_int)
            if d_candidates is None:
                continue

            state_c = ann_states[occ_idx]
            sign_c = ann_signs[occ_idx]
            for d in d_candidates:
                bit_d = orbital_bits[d]
                if (state_c & bit_d) == 0:
                    continue
                terms_cd = two_grouped_get((c_int, d))
                if terms_cd is None:
                    continue

                parity_d = (state_c & lower_masks[d]).bit_count() & 1
                sign_d = -1 if parity_d else 1
                state_cd = state_c ^ bit_d
                sign_cd = sign_c * sign_d

                for a, b, coeff in terms_cd:
                    bit_b = orbital_bits[b]
                    if (state_cd & bit_b) != 0:
                        continue
                    parity_b = (state_cd & lower_masks[b]).bit_count() & 1
                    sign_b = -1 if parity_b else 1
                    state_cdb = state_cd | bit_b

                    bit_a = orbital_bits[a]
                    if (state_cdb & bit_a) != 0:
                        continue
                    parity_a = (state_cdb & lower_masks[a]).bit_count() & 1
                    sign_a = -1 if parity_a else 1
                    state_final = state_cdb | bit_a

                    row = state_index_get(state_final)
                    if row is not None:
                        entries[(row, col)] += coeff * sign_cd * sign_b * sign_a

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
    operator_data: ManyBodyOperatorData | None = None,
) -> sp.csr_matrix:
    # Written with Codex 02-19-26.
    if operator_data is not None:
        h_numba = _build_many_body_hamiltonian_sparse_numba(
            basis=basis,
            operator_data=operator_data,
        )
        if h_numba is not None:
            if hermitize:
                h_numba = 0.5 * (h_numba + h_numba.getH())
            return h_numba

    rows, cols, data = _accumulate_many_body_entries(
        one_body=one_body,
        two_body=two_body,
        basis=basis,
        cutoff=cutoff,
        operator_data=operator_data,
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
    operator_data: ManyBodyOperatorData | None = None,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    operator_data_provided = operator_data is not None
    n_orb = int(basis.n_orbitals)
    if one_body.shape != (n_orb, n_orb):
        raise ValueError(
            f"one_body must have shape ({n_orb}, {n_orb}), got {one_body.shape}."
        )
    if two_body.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError(
            "two_body must have shape "
            f"({n_orb}, {n_orb}, {n_orb}, {n_orb}), got {two_body.shape}."
        )

    if operator_data is None:
        operator_data = prepare_many_body_operator_data(
            one_body=one_body,
            two_body=two_body,
            cutoff=cutoff,
        )

    # Numba is most beneficial when operator_data (and its compact cache) is reused.
    h_dense_numba = (
        _build_many_body_hamiltonian_dense_numba(
            basis=basis,
            operator_data=operator_data,
        )
        if operator_data_provided
        else None
    )
    if h_dense_numba is not None:
        h_dense = h_dense_numba
    else:
        rows, cols, data = _accumulate_many_body_entries(
            one_body=one_body,
            two_body=two_body,
            basis=basis,
            cutoff=cutoff,
            operator_data=operator_data,
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
