from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticeEmbedding2D:
    """Discrete 2D Bravais-cell embedding for a finite periodic lattice."""

    Lx: int
    Ly: int
    n_orbitals_per_cell: int
    site_index: np.ndarray
    site_to_cell: np.ndarray
    cell_to_bravais: np.ndarray
    supercell_matrix: np.ndarray
    kpoint_coefficients: np.ndarray


@dataclass(frozen=True)
class BlochBandData:
    """Bloch blocks and per-k diagonalization results."""

    lattice: LatticeEmbedding2D
    bloch_hamiltonians: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class ProjectedOrbitalBasis:
    """Projected single-particle basis built from selected Bloch bands."""

    lattice: LatticeEmbedding2D
    selected_bands: np.ndarray
    orbital_momenta: np.ndarray
    orbital_bands: np.ndarray
    site_amplitudes: np.ndarray
    one_body_matrix: np.ndarray


@dataclass(frozen=True)
class QuarticTermList:
    """List representation of c^dag_a c^dag_b c_d c_c operator coefficients."""

    create_1: np.ndarray
    create_2: np.ndarray
    annihilate_1: np.ndarray
    annihilate_2: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class ProjectedFermionHamiltonian:
    """Normal-ordered projected Hamiltonian data in a Bloch-derived basis."""

    lattice: LatticeEmbedding2D
    orbital_momenta: np.ndarray
    orbital_bands: np.ndarray
    one_body: np.ndarray
    two_body: np.ndarray


@dataclass
class ManyBodyBasis:
    """Fock basis (optionally momentum-resolved) in a fixed particle sector."""

    n_orbitals: int
    n_particles: int
    states: list[int]
    state_index: dict[int, int]
    momentum_sector: tuple[int, int] | None
