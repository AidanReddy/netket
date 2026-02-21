from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .types import ManyBodyBasis


def _occupied_orbitals_from_state(state: int, n_particles: int) -> np.ndarray:
    # Written with Codex 02-20-26.
    occ = np.empty(n_particles, dtype=np.int64)
    bits = int(state)
    idx = 0
    while bits:
        lsb = bits & -bits
        occ[idx] = int(lsb.bit_length() - 1)
        bits ^= lsb
        idx += 1
    if idx != n_particles:
        raise ValueError(
            f"State {state} has {idx} occupied orbitals, expected {n_particles}."
        )
    return occ


def enumerate_site_configurations(n_sites: int, n_particles: int) -> np.ndarray:
    # Written with Codex 02-20-26.
    if n_sites < 0:
        raise ValueError("n_sites must be non-negative.")
    if not (0 <= n_particles <= n_sites):
        raise ValueError("n_particles must satisfy 0 <= n_particles <= n_sites.")

    n_cfg = math.comb(n_sites, n_particles)
    out = np.empty((n_cfg, n_particles), dtype=np.int64)
    for idx, occ in enumerate(combinations(range(n_sites), n_particles)):
        out[idx] = occ
    return out


def expand_projected_state_to_site_basis(
    basis: ManyBodyBasis,
    coefficients: np.ndarray,
    site_amplitudes: np.ndarray,
    site_configurations: np.ndarray | None = None,
    cutoff: float = 1e-14,
    max_site_basis: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    coeffs = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if coeffs.size != len(basis.states):
        raise ValueError(
            f"coefficients length {coeffs.size} does not match basis size {len(basis.states)}."
        )

    phi = np.asarray(site_amplitudes, dtype=np.complex128)
    if phi.ndim != 2:
        raise ValueError("site_amplitudes must be a rank-2 array.")
    n_sites, n_orbitals = phi.shape
    if n_orbitals != int(basis.n_orbitals):
        raise ValueError(
            f"site_amplitudes has {n_orbitals} orbitals but basis has {basis.n_orbitals}."
        )

    n_particles = int(basis.n_particles)
    n_cfg_full = math.comb(n_sites, n_particles)
    if n_cfg_full > int(max_site_basis):
        raise ValueError(
            "Site-basis expansion is too large: "
            f"C({n_sites}, {n_particles})={n_cfg_full} > max_site_basis={max_site_basis}."
        )

    if site_configurations is None:
        site_cfg = enumerate_site_configurations(
            n_sites=n_sites,
            n_particles=n_particles,
        )
    else:
        site_cfg = np.asarray(site_configurations, dtype=np.int64)
        if site_cfg.ndim != 2 or site_cfg.shape[1] != n_particles:
            raise ValueError(
                "site_configurations must have shape (n_cfg, n_particles)."
            )
        if np.any(site_cfg < 0) or np.any(site_cfg >= n_sites):
            raise ValueError("site_configurations contains out-of-range site indices.")
        if np.any(np.diff(site_cfg, axis=1) <= 0):
            raise ValueError(
                "Each site configuration row must contain strictly increasing indices."
            )

    orbital_occ = np.empty((len(basis.states), n_particles), dtype=np.int64)
    for idx, state in enumerate(basis.states):
        orbital_occ[idx] = _occupied_orbitals_from_state(
            state=int(state),
            n_particles=n_particles,
        )

    # site_subspace[s, r, a] = <site_{s,r} | orbital_a>
    site_subspace = phi[site_cfg]
    psi_site = np.zeros(site_cfg.shape[0], dtype=np.complex128)

    for idx, amp in enumerate(coeffs):
        if abs(amp) <= cutoff:
            continue
        occ = orbital_occ[idx]
        # Determinants over all site configurations for this occupied-orbital set.
        dets = np.linalg.det(site_subspace[:, :, occ])
        psi_site += amp * dets

    return site_cfg, psi_site


def ordinary_density_observables_from_projected_state(
    basis: ManyBodyBasis,
    coefficients: np.ndarray,
    site_amplitudes: np.ndarray,
    site_configurations: np.ndarray | None = None,
    cutoff: float = 1e-14,
    max_site_basis: int = 2_000_000,
) -> dict[str, np.ndarray]:
    # Written with Codex 02-20-26.
    site_cfg, psi_site = expand_projected_state_to_site_basis(
        basis=basis,
        coefficients=coefficients,
        site_amplitudes=site_amplitudes,
        site_configurations=site_configurations,
        cutoff=cutoff,
        max_site_basis=max_site_basis,
    )

    n_sites = int(np.asarray(site_amplitudes).shape[0])
    prob = np.abs(psi_site) ** 2
    norm = float(prob.sum())
    if norm <= 0.0:
        raise RuntimeError("Expanded site-basis wavefunction has zero norm.")
    prob = prob / norm

    occ_matrix = np.zeros((site_cfg.shape[0], n_sites), dtype=np.float64)
    row_idx = np.arange(site_cfg.shape[0], dtype=np.int64)[:, None]
    occ_matrix[row_idx, site_cfg] = 1.0

    charge_density = np.einsum("s,si->i", prob, occ_matrix, optimize=True)
    corr_matrix = np.einsum("s,si,sj->ij", prob, occ_matrix, occ_matrix, optimize=True)

    return {
        "site_configurations": site_cfg,
        "psi_site": psi_site,
        "probabilities": prob,
        "charge_density": charge_density,
        "corr_matrix": corr_matrix,
    }
