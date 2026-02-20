from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .interactions import build_projected_hamiltonian_terms, make_density_density_terms
from .single_particle import (
    build_lattice_embedding_2d,
    build_projected_orbital_basis,
    compute_bloch_band_data,
)
from .types import LatticeEmbedding2D, ProjectedFermionHamiltonian, QuarticTermList


def build_haldane_lattice_embedding(Lx: int, Ly: int) -> LatticeEmbedding2D:
    # Written with Codex 02-19-26.
    import netket as nk

    graph = nk.graph.Honeycomb(extent=[Lx, Ly], pbc=True)

    def _site_index(x: int, y: int, sublattice: int) -> int:
        # Written with Codex 02-19-26.
        return int(graph.id_from_basis_coords([x % Lx, y % Ly, sublattice]))

    return build_lattice_embedding_2d(
        Lx=Lx,
        Ly=Ly,
        n_orbitals_per_cell=2,
        site_index_fn=_site_index,
    )


def build_haldane_real_space_terms(
    Lx: int,
    Ly: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    V1: float = 0.0,
) -> tuple[LatticeEmbedding2D, np.ndarray, QuarticTermList]:
    # Written with Codex 02-19-26.
    from ..haldane_model import build_haldane_one_body_matrix

    import netket as nk

    graph = nk.graph.Honeycomb(extent=[Lx, Ly], pbc=True)
    lattice = build_haldane_lattice_embedding(Lx=Lx, Ly=Ly)

    one_body_real_space = build_haldane_one_body_matrix(
        graph=graph,
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
    )

    if np.isclose(V1, 0.0):
        site_terms = make_density_density_terms([])
    else:
        weighted_pairs = [
            (int(i), int(j), np.complex128(V1)) for i, j in graph.edges()
        ]
        site_terms = make_density_density_terms(weighted_pairs)

    return lattice, one_body_real_space, site_terms


def build_haldane_projected_hamiltonian(
    Lx: int,
    Ly: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    selected_bands: Sequence[int],
    V1: float = 0.0,
) -> ProjectedFermionHamiltonian:
    # Written with Codex 02-19-26.
    lattice, one_body_real_space, site_terms = build_haldane_real_space_terms(
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
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
