from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .types import ProjectedFermionHamiltonian, ProjectedOrbitalBasis, QuarticTermList


def make_quartic_term_list(
    terms: Sequence[tuple[int, int, int, int, complex]],
) -> QuarticTermList:
    # Written with Codex 02-19-26.
    if len(terms) == 0:
        empty_i = np.asarray([], dtype=np.int64)
        empty_c = np.asarray([], dtype=np.complex128)
        return QuarticTermList(
            create_1=empty_i,
            create_2=empty_i,
            annihilate_1=empty_i,
            annihilate_2=empty_i,
            coefficients=empty_c,
        )

    create_1 = np.empty(len(terms), dtype=np.int64)
    create_2 = np.empty(len(terms), dtype=np.int64)
    annihilate_1 = np.empty(len(terms), dtype=np.int64)
    annihilate_2 = np.empty(len(terms), dtype=np.int64)
    coefficients = np.empty(len(terms), dtype=np.complex128)

    for idx, (a, b, c, d, coeff) in enumerate(terms):
        create_1[idx] = int(a)
        create_2[idx] = int(b)
        annihilate_1[idx] = int(c)
        annihilate_2[idx] = int(d)
        coefficients[idx] = np.complex128(coeff)

    return QuarticTermList(
        create_1=create_1,
        create_2=create_2,
        annihilate_1=annihilate_1,
        annihilate_2=annihilate_2,
        coefficients=coefficients,
    )


def make_density_density_terms(
    weighted_pairs: Iterable[tuple[int, int, complex]],
) -> QuarticTermList:
    # Written with Codex 02-19-26.
    terms: list[tuple[int, int, int, int, complex]] = []
    for i, j, value in weighted_pairs:
        ii = int(i)
        jj = int(j)
        if ii == jj:
            raise ValueError("Density-density terms require i != j.")
        terms.append((ii, jj, ii, jj, np.complex128(value)))
    return make_quartic_term_list(terms)


def project_quartic_terms_to_orbitals(
    site_terms: QuarticTermList,
    projected_basis: ProjectedOrbitalBasis,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    phi = projected_basis.site_amplitudes
    n_sites, n_orb = phi.shape
    n_terms = site_terms.coefficients.size

    if (
        np.any(site_terms.create_1 < 0)
        or np.any(site_terms.create_1 >= n_sites)
        or np.any(site_terms.create_2 < 0)
        or np.any(site_terms.create_2 >= n_sites)
        or np.any(site_terms.annihilate_1 < 0)
        or np.any(site_terms.annihilate_1 >= n_sites)
        or np.any(site_terms.annihilate_2 < 0)
        or np.any(site_terms.annihilate_2 >= n_sites)
    ):
        raise ValueError("Site index out of range in site quartic terms.")

    out = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.complex128)
    for t in range(n_terms):
        i = int(site_terms.create_1[t])
        j = int(site_terms.create_2[t])
        k = int(site_terms.annihilate_1[t])
        l = int(site_terms.annihilate_2[t])
        coeff = site_terms.coefficients[t]
        out += coeff * np.einsum(
            "a,b,c,d->abcd",
            phi[i].conjugate(),
            phi[j].conjugate(),
            phi[k],
            phi[l],
            optimize=True,
        )

    return out


def build_projected_hamiltonian_terms(
    projected_basis: ProjectedOrbitalBasis,
    site_terms: QuarticTermList | None = None,
) -> ProjectedFermionHamiltonian:
    # Written with Codex 02-19-26.
    n_orb = projected_basis.one_body_matrix.shape[0]
    if site_terms is None:
        two_body = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.complex128)
    else:
        two_body = project_quartic_terms_to_orbitals(site_terms, projected_basis)

    return ProjectedFermionHamiltonian(
        lattice=projected_basis.lattice,
        orbital_momenta=projected_basis.orbital_momenta,
        orbital_bands=projected_basis.orbital_bands,
        one_body=projected_basis.one_body_matrix,
        two_body=two_body,
    )


def extract_nonzero_quartic_terms(
    two_body: np.ndarray,
    cutoff: float = 1e-12,
    orbital_momenta: np.ndarray | None = None,
    lattice_shape: tuple[int, int] | None = None,
    enforce_momentum_conservation: bool = False,
) -> QuarticTermList:
    # Written with Codex 02-19-26.
    tensor = np.asarray(two_body, dtype=np.complex128)
    if tensor.ndim != 4:
        raise ValueError("two_body must be a rank-4 tensor.")
    if not (
        tensor.shape[0] == tensor.shape[1]
        and tensor.shape[0] == tensor.shape[2]
        and tensor.shape[0] == tensor.shape[3]
    ):
        raise ValueError("two_body must have shape (M, M, M, M).")

    nz = np.argwhere(np.abs(tensor) > cutoff)
    if nz.shape[0] == 0:
        return make_quartic_term_list([])

    if enforce_momentum_conservation:
        if orbital_momenta is None or lattice_shape is None:
            raise ValueError(
                "Need orbital_momenta and lattice_shape when "
                "enforce_momentum_conservation=True."
            )
        Lx, Ly = lattice_shape
        keep_mask = np.empty(nz.shape[0], dtype=bool)
        for idx in range(nz.shape[0]):
            a, b, c, d = nz[idx]
            dkx = (
                int(orbital_momenta[a, 0])
                + int(orbital_momenta[b, 0])
                - int(orbital_momenta[c, 0])
                - int(orbital_momenta[d, 0])
            ) % Lx
            dky = (
                int(orbital_momenta[a, 1])
                + int(orbital_momenta[b, 1])
                - int(orbital_momenta[c, 1])
                - int(orbital_momenta[d, 1])
            ) % Ly
            keep_mask[idx] = (dkx == 0) and (dky == 0)
        nz = nz[keep_mask]
        if nz.shape[0] == 0:
            return make_quartic_term_list([])

    terms = []
    for a, b, c, d in nz:
        terms.append((int(a), int(b), int(c), int(d), tensor[a, b, c, d]))
    return make_quartic_term_list(terms)
