from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import fold_to_shortest_k
from ..mote2_three_orbital import (
    _mote2_vectors,
    mote2_three_orbital_bloch_hamiltonian,
    mote2_three_orbital_reciprocal_vectors,
)
from .interactions import build_projected_hamiltonian_terms, make_density_density_terms
from .single_particle import (
    build_lattice_embedding_2d,
    build_projected_orbital_basis,
    compute_bloch_band_data,
)
from .types import (
    LatticeEmbedding2D,
    ProjectedFermionHamiltonian,
    QuarticTermList,
)


@dataclass(frozen=True)
class _Mote2SupercellData:
    canonical_lx: int
    canonical_ly: int
    cell_to_bravais: np.ndarray
    supercell_matrix: np.ndarray
    kpoint_coefficients: np.ndarray


def _as_integer_supercell_matrix(
    supercell_matrix: Sequence[Sequence[int]],
) -> np.ndarray:
    # Written with Codex 02-21-26.
    matrix_raw = np.asarray(supercell_matrix)
    if matrix_raw.shape != (2, 2):
        raise ValueError(
            "supercell_matrix must have shape (2, 2), "
            f"got {matrix_raw.shape}."
        )
    matrix_int = np.rint(matrix_raw).astype(np.int64)
    if not np.allclose(matrix_raw, matrix_int):
        raise ValueError("supercell_matrix must contain integer entries.")
    det_matrix = int(matrix_int[0, 0] * matrix_int[1, 1] - matrix_int[0, 1] * matrix_int[1, 0])
    if det_matrix == 0:
        raise ValueError("supercell_matrix must be invertible over integers (det != 0).")
    return matrix_int


def _as_integer_supercell_basis(
    supercell_basis: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    if supercell_basis is None:
        return np.eye(2, dtype=np.int64)
    return _as_integer_supercell_matrix(supercell_basis)


def _resolve_mote2_supercell_matrix(
    Lx: int | None,
    Ly: int | None,
    supercell_basis: Sequence[Sequence[int]] | None,
    supercell_matrix: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    if supercell_matrix is not None:
        # If the matrix is given, it defines the supercell completely.
        return _as_integer_supercell_matrix(supercell_matrix)

    if Lx is None or Ly is None:
        raise ValueError(
            "supercell_matrix is required when Lx/Ly are not provided."
        )
    if int(Lx) <= 0 or int(Ly) <= 0:
        raise ValueError("Lx and Ly must both be positive.")

    basis_int = _as_integer_supercell_basis(supercell_basis)
    scale = np.array([[int(Lx), 0], [0, int(Ly)]], dtype=np.int64)
    return basis_int @ scale


def _inverse_unimodular_2x2(matrix: np.ndarray) -> np.ndarray:
    # Written with Codex 02-21-26.
    mat = np.asarray(matrix, dtype=np.int64)
    if mat.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {mat.shape}.")
    det = int(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    if abs(det) != 1:
        raise ValueError(f"matrix must be unimodular with det=+/-1, got det={det}.")
    inv = np.array(
        [
            [mat[1, 1], -mat[0, 1]],
            [-mat[1, 0], mat[0, 0]],
        ],
        dtype=np.int64,
    )
    if det == -1:
        inv = -inv
    return inv


def _smith_normal_form_2x2(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    mat = np.asarray(matrix, dtype=np.int64)
    if mat.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {mat.shape}.")
    det = int(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    if det == 0:
        raise ValueError("matrix must be full-rank (det != 0).")

    a = mat.astype(object)
    u = np.eye(2, dtype=object)
    v = np.eye(2, dtype=object)

    def _swap_rows() -> None:
        # Written with Codex 02-21-26.
        nonlocal a, u
        a[[0, 1], :] = a[[1, 0], :]
        u[[0, 1], :] = u[[1, 0], :]

    def _swap_cols() -> None:
        # Written with Codex 02-21-26.
        nonlocal a, v
        a[:, [0, 1]] = a[:, [1, 0]]
        v[:, [0, 1]] = v[:, [1, 0]]

    def _row_add(src: int, dst: int, factor: int) -> None:
        # Written with Codex 02-21-26.
        nonlocal a, u
        a[dst, :] = a[dst, :] + factor * a[src, :]
        u[dst, :] = u[dst, :] + factor * u[src, :]

    def _col_add(src: int, dst: int, factor: int) -> None:
        # Written with Codex 02-21-26.
        nonlocal a, v
        a[:, dst] = a[:, dst] + factor * a[:, src]
        v[:, dst] = v[:, dst] + factor * v[:, src]

    def _row_mul(row: int, factor: int) -> None:
        # Written with Codex 02-21-26.
        nonlocal a, u
        a[row, :] = factor * a[row, :]
        u[row, :] = factor * u[row, :]

    def _reduce_first_column() -> None:
        # Written with Codex 02-21-26.
        guard = 0
        while a[1, 0] != 0:
            guard += 1
            if guard > 200:
                raise RuntimeError(
                    "Smith-normal-form column reduction did not converge for matrix "
                    f"{mat.tolist()}."
                )
            if a[0, 0] == 0:
                _swap_rows()
                continue
            quotient = int(a[1, 0] // a[0, 0])
            _row_add(src=0, dst=1, factor=-quotient)
            if a[1, 0] != 0:
                _swap_rows()

    def _reduce_first_row() -> None:
        # Written with Codex 02-21-26.
        guard = 0
        while a[0, 1] != 0:
            guard += 1
            if guard > 200:
                raise RuntimeError(
                    "Smith-normal-form row reduction did not converge for matrix "
                    f"{mat.tolist()}."
                )
            if a[0, 0] == 0:
                _swap_cols()
                continue
            quotient = int(a[0, 1] // a[0, 0])
            _col_add(src=0, dst=1, factor=-quotient)
            if a[0, 1] != 0:
                _swap_cols()

    for _ in range(300):
        if a[0, 0] == 0:
            if a[1, 0] != 0:
                _swap_rows()
            elif a[0, 1] != 0:
                _swap_cols()
            elif a[1, 1] != 0:
                _swap_rows()
                _swap_cols()

        _reduce_first_column()
        _reduce_first_row()

        if a[0, 0] == 0:
            continue
        if a[1, 0] == 0 and a[0, 1] == 0:
            if int(a[1, 1]) % int(a[0, 0]) == 0:
                break
            # Force coupling with the lower-right entry, then continue Euclidean reduction.
            _col_add(src=1, dst=0, factor=1)
    else:
        raise RuntimeError(
            "Smith-normal-form reduction did not converge for matrix "
            f"{mat.tolist()}."
        )

    if a[0, 0] < 0:
        _row_mul(row=0, factor=-1)
    if a[1, 1] < 0:
        _row_mul(row=1, factor=-1)
    if abs(int(a[0, 0])) > abs(int(a[1, 1])):
        _swap_rows()
        _swap_cols()

    d = np.asarray(a, dtype=np.int64)
    u_arr = np.asarray(u, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    return d, u_arr, v_arr


def _build_mote2_supercell_data(
    supercell_matrix: Sequence[Sequence[int]],
) -> _Mote2SupercellData:
    # Written with Codex 02-21-26.
    supercell_matrix_int = _as_integer_supercell_matrix(supercell_matrix)

    smith_d, smith_u, _ = _smith_normal_form_2x2(supercell_matrix_int)
    canonical_lx = int(smith_d[0, 0])
    canonical_ly = int(smith_d[1, 1])
    if canonical_lx <= 0 or canonical_ly <= 0:
        raise RuntimeError(
            "Invalid Smith-normal-form diagonal entries "
            f"{smith_d.diagonal().tolist()} for supercell matrix {supercell_matrix_int.tolist()}."
        )

    smith_u_inv = _inverse_unimodular_2x2(smith_u)
    cell_to_bravais = np.empty((canonical_lx, canonical_ly, 2), dtype=np.int64)
    for x in range(canonical_lx):
        for y in range(canonical_ly):
            cell_to_bravais[x, y] = smith_u_inv @ np.array([x, y], dtype=np.int64)

    kpoint_coefficients = np.empty((canonical_lx, canonical_ly, 2), dtype=np.float64)
    smith_u_t = np.asarray(smith_u, dtype=np.float64).T
    for kx in range(canonical_lx):
        for ky in range(canonical_ly):
            kpoint_coefficients[kx, ky] = smith_u_t @ np.array(
                [float(kx) / float(canonical_lx), float(ky) / float(canonical_ly)],
                dtype=np.float64,
            )

    return _Mote2SupercellData(
        canonical_lx=canonical_lx,
        canonical_ly=canonical_ly,
        cell_to_bravais=cell_to_bravais,
        supercell_matrix=supercell_matrix_int,
        kpoint_coefficients=kpoint_coefficients,
    )


def _build_mote2_three_orbital_lattice_embedding_from_supercell_data(
    supercell_data: _Mote2SupercellData,
) -> LatticeEmbedding2D:
    # Written with Codex 02-21-26.
    n_orbitals_per_cell = 3
    canonical_lx = int(supercell_data.canonical_lx)
    canonical_ly = int(supercell_data.canonical_ly)
    return build_lattice_embedding_2d(
        Lx=canonical_lx,
        Ly=canonical_ly,
        n_orbitals_per_cell=n_orbitals_per_cell,
        site_index_fn=lambda x, y, orb: (
            ((x % canonical_lx) * canonical_ly + (y % canonical_ly)) * n_orbitals_per_cell + orb
        ),
        cell_to_bravais=supercell_data.cell_to_bravais,
        supercell_matrix=supercell_data.supercell_matrix,
        kpoint_coefficients=supercell_data.kpoint_coefficients,
    )


def build_mote2_three_orbital_equal_aspect_27_supercell_basis() -> np.ndarray:
    # Written with Codex 02-21-26.
    # Second-nearest-neighbor basis vectors:
    # u1 = a1 - a2, u2 = 2*a1 + a2.
    return np.array([[1, 2], [-1, 1]], dtype=np.int64)


def build_mote2_three_orbital_equal_aspect_27_supercell_matrix() -> np.ndarray:
    # Written with Codex 02-21-26.
    basis_2nn = build_mote2_three_orbital_equal_aspect_27_supercell_basis()
    return basis_2nn @ np.array([[3, 0], [0, 3]], dtype=np.int64)


def build_mote2_three_orbital_lattice_embedding(
    Lx: int | None = None,
    Ly: int | None = None,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> LatticeEmbedding2D:
    # Written with Codex 02-20-26.
    supercell_matrix_eff = _resolve_mote2_supercell_matrix(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    supercell_data = _build_mote2_supercell_data(
        supercell_matrix=supercell_matrix_eff,
    )
    return _build_mote2_three_orbital_lattice_embedding_from_supercell_data(supercell_data)


def build_mote2_three_orbital_supercell_vectors(
    Lx: int | None = None,
    Ly: int | None = None,
    a_m: float = 1.0,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    a1, a2, _, _, _ = _mote2_vectors(float(a_m))
    supercell_t1 = (
        float(lattice.supercell_matrix[0, 0]) * np.asarray(a1, dtype=float)
        + float(lattice.supercell_matrix[1, 0]) * np.asarray(a2, dtype=float)
    )
    supercell_t2 = (
        float(lattice.supercell_matrix[0, 1]) * np.asarray(a1, dtype=float)
        + float(lattice.supercell_matrix[1, 1]) * np.asarray(a2, dtype=float)
    )
    return supercell_t1, supercell_t2


def build_mote2_three_orbital_supercell_bravais_positions(
    Lx: int | None = None,
    Ly: int | None = None,
    a_m: float = 1.0,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    a1, a2, _, _, _ = _mote2_vectors(float(a_m))
    positions = np.empty((lattice.Lx * lattice.Ly, 2), dtype=np.float64)
    idx = 0
    for x in range(lattice.Lx):
        for y in range(lattice.Ly):
            coeff = lattice.cell_to_bravais[x, y]
            positions[idx] = (
                float(coeff[0]) * np.asarray(a1, dtype=float)
                + float(coeff[1]) * np.asarray(a2, dtype=float)
            )
            idx += 1
    return positions


def build_mote2_three_orbital_supercell_k_mesh(
    Lx: int | None = None,
    Ly: int | None = None,
    a_m: float = 1.0,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
    fold_to_bz: bool = True,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    kpts = np.empty((lattice.Lx * lattice.Ly, 2), dtype=np.float64)
    idx = 0
    for kx in range(lattice.Lx):
        for ky in range(lattice.Ly):
            coeff = lattice.kpoint_coefficients[kx, ky]
            kvec = coeff[0] * b1 + coeff[1] * b2
            if fold_to_bz:
                kvec = fold_to_shortest_k(kvec, b1, b2)
            kpts[idx] = kvec
            idx += 1

    if not fold_to_bz:
        return kpts

    keys = np.round(kpts, decimals=12)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return kpts[np.sort(unique_idx)]


def _mote2_discrete_bloch_blocks(
    kpoint_coefficients: np.ndarray,
    delta: float,
    ez: float,
    t_th1: float,
    t_hh1: float,
    t_th2: float,
    t_hh3: float,
    t_tt1: float,
    a_m: float,
    ph_conj: bool,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    coeffs = np.asarray(kpoint_coefficients, dtype=np.float64)
    if coeffs.ndim != 3 or coeffs.shape[2] != 2:
        raise ValueError(
            "kpoint_coefficients must have shape (Lx, Ly, 2), "
            f"got {coeffs.shape}."
        )

    Lx = int(coeffs.shape[0])
    Ly = int(coeffs.shape[1])
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    h_bloch = np.empty((Lx, Ly, 3, 3), dtype=np.complex128)

    for kx_idx in range(Lx):
        for ky_idx in range(Ly):
            coeff = coeffs[kx_idx, ky_idx]
            kvec = coeff[0] * b1 + coeff[1] * b2
            h_bloch[kx_idx, ky_idx] = mote2_three_orbital_bloch_hamiltonian(
                kx=float(kvec[0]),
                ky=float(kvec[1]),
                delta=float(delta),
                ez=float(ez),
                t_th1=float(t_th1),
                t_hh1=float(t_hh1),
                t_th2=float(t_th2),
                t_hh3=float(t_hh3),
                t_tt1=float(t_tt1),
                a_m=float(a_m),
                ph_conj=bool(ph_conj),
            )

    return h_bloch


def build_mote2_three_orbital_one_body_matrix(
    Lx: int | None = None,
    Ly: int | None = None,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell

    h_bloch = _mote2_discrete_bloch_blocks(
        kpoint_coefficients=lattice.kpoint_coefficients,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
    )
    h_bloch = _mote2_embedding_gauge_transform_blocks(
        h_bloch=h_bloch,
        a_m=a_m,
        kpoint_coefficients=lattice.kpoint_coefficients,
    )
    hop = np.fft.ifftn(h_bloch, axes=(0, 1))

    Lx_c = lattice.Lx
    Ly_c = lattice.Ly
    one_body = np.empty((n_sites, n_sites), dtype=np.complex128)
    for i in range(n_sites):
        xi, yi, orbi = lattice.site_to_cell[i]
        for j in range(n_sites):
            xj, yj, orbj = lattice.site_to_cell[j]
            dx = (int(xj) - int(xi)) % Lx_c
            dy = (int(yj) - int(yi)) % Ly_c
            one_body[i, j] = hop[dx, dy, int(orbi), int(orbj)]

    one_body = 0.5 * (one_body + np.conjugate(one_body.T))
    return one_body


def _mote2_orbital_offsets(a_m: float) -> np.ndarray:
    # Written with Codex 02-20-26.
    # Basis order in the Bloch Hamiltonian is (MX, MM, XM).
    _, _, _, _, u_vectors = _mote2_vectors(float(a_m))
    u0 = np.asarray(u_vectors[0], dtype=float)
    return np.asarray(
        (
            u0,                # MX
            np.zeros(2),       # MM
            -u0,               # XM
        ),
        dtype=float,
    )


def _mote2_embedding_gauge_transform_blocks(
    h_bloch: np.ndarray,
    a_m: float,
    kpoint_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    blocks = np.asarray(h_bloch, dtype=np.complex128)
    if blocks.ndim != 4 or blocks.shape[2:] != (3, 3):
        raise ValueError(
            "h_bloch must have shape (Lx, Ly, 3, 3), "
            f"got {blocks.shape}."
        )

    Lx = int(blocks.shape[0])
    Ly = int(blocks.shape[1])
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    if kpoint_coefficients is None:
        coeffs = np.empty((Lx, Ly, 2), dtype=np.float64)
        for kx in range(Lx):
            for ky in range(Ly):
                coeffs[kx, ky, 0] = float(kx) / float(Lx)
                coeffs[kx, ky, 1] = float(ky) / float(Ly)
    else:
        coeffs = np.asarray(kpoint_coefficients, dtype=np.float64)
        if coeffs.shape != (Lx, Ly, 2):
            raise ValueError(
                "kpoint_coefficients must have shape "
                f"({Lx}, {Ly}, 2), got {coeffs.shape}."
            )

    orbital_offsets = _mote2_orbital_offsets(float(a_m))
    transformed = np.empty_like(blocks)
    for kx in range(Lx):
        for ky in range(Ly):
            coeff = coeffs[kx, ky]
            kvec = coeff[0] * b1 + coeff[1] * b2
            gauge = np.exp(-1j * (orbital_offsets @ kvec))
            transformed[kx, ky] = (
                gauge[:, None]
                * blocks[kx, ky]
                * np.conjugate(gauge[None, :])
            )
    return transformed


def _mote2_site_positions(
    lattice: LatticeEmbedding2D,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    a1, a2, _, _, _ = _mote2_vectors(float(a_m))
    orbital_offsets = _mote2_orbital_offsets(float(a_m))

    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell
    positions = np.empty((n_sites, 2), dtype=float)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        cell_bravais = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = (
            float(cell_bravais[0]) * np.asarray(a1, dtype=float)
            + float(cell_bravais[1]) * np.asarray(a2, dtype=float)
            + orbital_offsets[int(orb)]
        )

    supercell_t1 = (
        float(lattice.supercell_matrix[0, 0]) * np.asarray(a1, dtype=float)
        + float(lattice.supercell_matrix[1, 0]) * np.asarray(a2, dtype=float)
    )
    supercell_t2 = (
        float(lattice.supercell_matrix[0, 1]) * np.asarray(a1, dtype=float)
        + float(lattice.supercell_matrix[1, 1]) * np.asarray(a2, dtype=float)
    )
    return positions, supercell_t1, supercell_t2


def _minimum_image_distance_2d(
    delta: np.ndarray,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> float:
    # Written with Codex 02-20-26.
    best = np.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            trial = delta + float(n1) * supercell_t1 + float(n2) * supercell_t2
            dist = float(np.linalg.norm(trial))
            if dist < best:
                best = dist
    return best


def _mote2_true_nearest_neighbor_site_pairs(
    lattice: LatticeEmbedding2D,
    a_m: float,
    distance_tol: float = 1e-10,
) -> list[tuple[int, int]]:
    # Written with Codex 02-20-26.
    positions, supercell_t1, supercell_t2 = _mote2_site_positions(
        lattice=lattice,
        a_m=a_m,
    )
    n_sites = positions.shape[0]

    d_min = np.inf
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            d_ij = _minimum_image_distance_2d(
                delta=positions[j] - positions[i],
                supercell_t1=supercell_t1,
                supercell_t2=supercell_t2,
            )
            if d_ij > distance_tol and d_ij < d_min:
                d_min = d_ij

    if not np.isfinite(d_min):
        raise RuntimeError("Could not determine nearest-neighbor distance for MoTe2 embedding.")

    eps = max(float(distance_tol), 1e-8 * d_min)
    pairs: list[tuple[int, int]] = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            d_ij = _minimum_image_distance_2d(
                delta=positions[j] - positions[i],
                supercell_t1=supercell_t1,
                supercell_t2=supercell_t2,
            )
            if abs(d_ij - d_min) <= eps:
                pairs.append((int(i), int(j)))

    return pairs


def _mote2_true_nearest_neighbor_density_pairs(
    lattice: LatticeEmbedding2D,
    a_m: float,
    V1: float,
) -> QuarticTermList:
    # Written with Codex 02-20-26.
    if np.isclose(V1, 0.0):
        return make_density_density_terms([])

    weighted_pairs: list[tuple[int, int, np.complex128]] = []
    for site_i, site_j in _mote2_true_nearest_neighbor_site_pairs(
        lattice=lattice,
        a_m=a_m,
    ):
        orb_i = int(lattice.site_to_cell[site_i, 2])
        orb_j = int(lattice.site_to_cell[site_j, 2])
        # Basis order is (MX, MM, XM); keep only MX-MM and XM-MM bonds.
        if orb_i != 1 and orb_j != 1:
            continue
        weighted_pairs.append((site_i, site_j, np.complex128(V1)))

    return make_density_density_terms(weighted_pairs)


def build_mote2_three_orbital_real_space_terms(
    Lx: int | None = None,
    Ly: int | None = None,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
    V1: float = 0.0,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> tuple[LatticeEmbedding2D, np.ndarray, QuarticTermList]:
    # Written with Codex 02-20-26.
    lattice = build_mote2_three_orbital_lattice_embedding(
        Lx=Lx,
        Ly=Ly,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    one_body_real_space = build_mote2_three_orbital_one_body_matrix(
        Lx=Lx,
        Ly=Ly,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )
    site_terms = _mote2_true_nearest_neighbor_density_pairs(
        lattice=lattice,
        a_m=a_m,
        V1=V1,
    )
    return lattice, one_body_real_space, site_terms


def build_mote2_three_orbital_projected_hamiltonian(
    selected_bands: Sequence[int],
    Lx: int | None = None,
    Ly: int | None = None,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
    V1: float = 0.0,
    supercell_basis: Sequence[Sequence[int]] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> ProjectedFermionHamiltonian:
    # Written with Codex 02-20-26.
    lattice, one_body_real_space, site_terms = build_mote2_three_orbital_real_space_terms(
        Lx=Lx,
        Ly=Ly,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
        V1=V1,
        supercell_basis=supercell_basis,
        supercell_matrix=supercell_matrix,
    )

    band_data = compute_bloch_band_data(
        one_body_real_space=one_body_real_space,
        lattice=lattice,
    )
    projected_basis = build_projected_orbital_basis(
        one_body_real_space=one_body_real_space,
        band_data=band_data,
        selected_bands=selected_bands,
    )
    return build_projected_hamiltonian_terms(
        projected_basis=projected_basis,
        site_terms=site_terms,
    )
