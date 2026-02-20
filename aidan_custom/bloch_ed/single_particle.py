from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .types import BlochBandData, LatticeEmbedding2D, ProjectedOrbitalBasis


def build_lattice_embedding_2d(
    Lx: int,
    Ly: int,
    n_orbitals_per_cell: int,
    site_index_fn: Callable[[int, int, int], int],
) -> LatticeEmbedding2D:
    # Written with Codex 02-19-26.
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must both be positive.")
    if n_orbitals_per_cell <= 0:
        raise ValueError("n_orbitals_per_cell must be positive.")

    site_index = np.empty((Lx, Ly, n_orbitals_per_cell), dtype=np.int64)
    for x in range(Lx):
        for y in range(Ly):
            for orb in range(n_orbitals_per_cell):
                site_index[x, y, orb] = int(site_index_fn(x, y, orb))

    n_sites = Lx * Ly * n_orbitals_per_cell
    flat = site_index.reshape(-1)
    sorted_ids = np.sort(flat)
    if sorted_ids.size != n_sites or not np.array_equal(sorted_ids, np.arange(n_sites)):
        raise ValueError(
            "site_index_fn must generate a one-to-one mapping to [0, n_sites)."
        )

    site_to_cell = np.empty((n_sites, 3), dtype=np.int64)
    for x in range(Lx):
        for y in range(Ly):
            for orb in range(n_orbitals_per_cell):
                site = int(site_index[x, y, orb])
                site_to_cell[site, 0] = x
                site_to_cell[site, 1] = y
                site_to_cell[site, 2] = orb

    return LatticeEmbedding2D(
        Lx=Lx,
        Ly=Ly,
        n_orbitals_per_cell=n_orbitals_per_cell,
        site_index=site_index,
        site_to_cell=site_to_cell,
    )


def _translation_averaged_hopping_table(
    one_body_real_space: np.ndarray,
    lattice: LatticeEmbedding2D,
    tol: float,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    h_real = np.asarray(one_body_real_space, dtype=np.complex128)
    n_sites = lattice.Lx * lattice.Ly * lattice.n_orbitals_per_cell
    if h_real.shape != (n_sites, n_sites):
        raise ValueError(
            "one_body_real_space must have shape "
            f"({n_sites}, {n_sites}), got {h_real.shape}."
        )

    Lx = lattice.Lx
    Ly = lattice.Ly
    n_orb = lattice.n_orbitals_per_cell
    hop = np.zeros((Lx, Ly, n_orb, n_orb), dtype=np.complex128)

    for i in range(n_sites):
        xi, yi, orbi = lattice.site_to_cell[i]
        row = h_real[i]
        nonzero_j = np.flatnonzero(np.abs(row) > tol)
        for j in nonzero_j:
            coeff = row[j]
            xj, yj, orbj = lattice.site_to_cell[j]
            dx = (int(xj) - int(xi)) % Lx
            dy = (int(yj) - int(yi)) % Ly
            hop[dx, dy, int(orbi), int(orbj)] += coeff

    hop /= float(Lx * Ly)
    return hop


def compute_bloch_band_data(
    one_body_real_space: np.ndarray,
    lattice: LatticeEmbedding2D,
    tol: float = 1e-12,
) -> BlochBandData:
    # Written with Codex 02-19-26.
    hop = _translation_averaged_hopping_table(one_body_real_space, lattice, tol=tol)

    # Discrete Fourier transform over cell displacements.
    h_bloch = np.fft.fftn(hop, axes=(0, 1))
    h_bloch = 0.5 * (h_bloch + np.conjugate(np.swapaxes(h_bloch, 2, 3)))

    Lx = lattice.Lx
    Ly = lattice.Ly
    n_orb = lattice.n_orbitals_per_cell
    evals = np.empty((Lx, Ly, n_orb), dtype=np.float64)
    evecs = np.empty((Lx, Ly, n_orb, n_orb), dtype=np.complex128)

    for kx in range(Lx):
        for ky in range(Ly):
            vals_k, vecs_k = np.linalg.eigh(h_bloch[kx, ky])
            evals[kx, ky] = np.real(vals_k)
            evecs[kx, ky] = vecs_k

    return BlochBandData(
        lattice=lattice,
        bloch_hamiltonians=h_bloch,
        eigenvalues=evals,
        eigenvectors=evecs,
    )


def build_projected_orbital_basis(
    one_body_real_space: np.ndarray,
    band_data: BlochBandData,
    selected_bands: Sequence[int],
) -> ProjectedOrbitalBasis:
    # Written with Codex 02-19-26.
    bands = np.asarray(selected_bands, dtype=np.int64).reshape(-1)
    if bands.size == 0:
        raise ValueError("selected_bands must not be empty.")

    lattice = band_data.lattice
    n_orb = lattice.n_orbitals_per_cell
    if np.any(bands < 0) or np.any(bands >= n_orb):
        raise ValueError(
            f"selected_bands must be in [0, {n_orb}), got {bands.tolist()}."
        )

    Lx = lattice.Lx
    Ly = lattice.Ly
    n_cells = Lx * Ly
    n_sites = n_cells * n_orb
    n_proj = n_cells * bands.size

    h_real = np.asarray(one_body_real_space, dtype=np.complex128)
    if h_real.shape != (n_sites, n_sites):
        raise ValueError(
            "one_body_real_space must have shape "
            f"({n_sites}, {n_sites}), got {h_real.shape}."
        )

    phase_x = np.exp(
        -2j * np.pi * np.outer(np.arange(Lx, dtype=np.float64), np.arange(Lx, dtype=np.float64)) / float(Lx)
    )
    phase_y = np.exp(
        -2j * np.pi * np.outer(np.arange(Ly, dtype=np.float64), np.arange(Ly, dtype=np.float64)) / float(Ly)
    )
    norm = np.sqrt(float(n_cells))

    site_amplitudes = np.zeros((n_sites, n_proj), dtype=np.complex128)
    orbital_momenta = np.empty((n_proj, 2), dtype=np.int64)
    orbital_bands = np.empty((n_proj,), dtype=np.int64)

    idx = 0
    for kx in range(Lx):
        for ky in range(Ly):
            vecs_k = band_data.eigenvectors[kx, ky]
            for band in bands:
                orbital_momenta[idx, 0] = kx
                orbital_momenta[idx, 1] = ky
                orbital_bands[idx] = int(band)
                for x in range(Lx):
                    px = phase_x[kx, x]
                    for y in range(Ly):
                        phase = (px * phase_y[ky, y]) / norm
                        for orb in range(n_orb):
                            site = int(lattice.site_index[x, y, orb])
                            site_amplitudes[site, idx] = phase * vecs_k[orb, int(band)]
                idx += 1

    one_body_projected = site_amplitudes.conj().T @ h_real @ site_amplitudes
    one_body_projected = 0.5 * (
        one_body_projected + np.conjugate(one_body_projected.T)
    )

    return ProjectedOrbitalBasis(
        lattice=lattice,
        selected_bands=bands,
        orbital_momenta=orbital_momenta,
        orbital_bands=orbital_bands,
        site_amplitudes=site_amplitudes,
        one_body_matrix=one_body_projected,
    )
