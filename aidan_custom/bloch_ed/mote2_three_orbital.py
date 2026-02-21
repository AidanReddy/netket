from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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


def build_mote2_three_orbital_lattice_embedding(Lx: int, Ly: int) -> LatticeEmbedding2D:
    # Written with Codex 02-20-26.
    n_orbitals_per_cell = 3
    return build_lattice_embedding_2d(
        Lx=Lx,
        Ly=Ly,
        n_orbitals_per_cell=n_orbitals_per_cell,
        site_index_fn=lambda x, y, orb: (
            ((x % Lx) * Ly + (y % Ly)) * n_orbitals_per_cell + orb
        ),
    )


def _mote2_discrete_bloch_blocks(
    Lx: int,
    Ly: int,
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
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    h_bloch = np.empty((Lx, Ly, 3, 3), dtype=np.complex128)

    for kx_idx in range(Lx):
        for ky_idx in range(Ly):
            kvec = (
                (float(kx_idx) / float(Lx)) * b1
                + (float(ky_idx) / float(Ly)) * b2
            )
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
    Lx: int,
    Ly: int,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    lattice = build_mote2_three_orbital_lattice_embedding(Lx=Lx, Ly=Ly)
    n_sites = Lx * Ly * lattice.n_orbitals_per_cell

    h_bloch = _mote2_discrete_bloch_blocks(
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
    )
    h_bloch = _mote2_embedding_gauge_transform_blocks(
        h_bloch=h_bloch,
        a_m=a_m,
    )
    hop = np.fft.ifftn(h_bloch, axes=(0, 1))

    one_body = np.empty((n_sites, n_sites), dtype=np.complex128)
    for i in range(n_sites):
        xi, yi, orbi = lattice.site_to_cell[i]
        for j in range(n_sites):
            xj, yj, orbj = lattice.site_to_cell[j]
            dx = (int(xj) - int(xi)) % Lx
            dy = (int(yj) - int(yi)) % Ly
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
    orbital_offsets = _mote2_orbital_offsets(float(a_m))
    transformed = np.empty_like(blocks)
    for kx in range(Lx):
        for ky in range(Ly):
            kvec = (
                (float(kx) / float(Lx)) * b1
                + (float(ky) / float(Ly)) * b2
            )
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
        positions[site] = (
            float(x) * np.asarray(a1, dtype=float)
            + float(y) * np.asarray(a2, dtype=float)
            + orbital_offsets[int(orb)]
        )

    supercell_t1 = float(lattice.Lx) * np.asarray(a1, dtype=float)
    supercell_t2 = float(lattice.Ly) * np.asarray(a2, dtype=float)
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
    Lx: int,
    Ly: int,
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
) -> tuple[LatticeEmbedding2D, np.ndarray, QuarticTermList]:
    # Written with Codex 02-20-26.
    lattice = build_mote2_three_orbital_lattice_embedding(Lx=Lx, Ly=Ly)
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
    )
    site_terms = _mote2_true_nearest_neighbor_density_pairs(
        lattice=lattice,
        a_m=a_m,
        V1=V1,
    )
    return lattice, one_body_real_space, site_terms


def build_mote2_three_orbital_projected_hamiltonian(
    Lx: int,
    Ly: int,
    selected_bands: Sequence[int],
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
