from __future__ import annotations

import numpy as np

from .geometry import fold_to_shortest_k, reciprocal_from_basis


def minimum_image_translations(
    basis_vectors: np.ndarray,
    Lx: int,
    Ly: int,
    pbc: np.ndarray,
) -> np.ndarray:
    # Written with Codex 02-18-26.
    pbc = np.asarray(pbc, dtype=bool).reshape(-1)
    if pbc.size == 1:
        pbc = np.repeat(pbc, 2)

    shift_n1 = (-1, 0, 1) if pbc[0] else (0,)
    shift_n2 = (-1, 0, 1) if pbc[1] else (0,)

    translations = [
        (n1 * Lx) * basis_vectors[0] + (n2 * Ly) * basis_vectors[1]
        for n1 in shift_n1
        for n2 in shift_n2
    ]
    return np.asarray(translations, dtype=float)


def pair_correlation_cartesian(
    samples: np.ndarray,
    positions: np.ndarray,
    basis_vectors: np.ndarray,
    pbc: np.ndarray,
    Lx: int,
    Ly: int,
    n_fermions: int,
) -> dict:
    # Written with Codex 02-18-26.
    n_samples = int(samples.shape[0])
    n_sites = int(samples.shape[1])

    charge_density = samples.mean(axis=0)
    corr_matrix = (samples.T @ samples) / n_samples

    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    translations = minimum_image_translations(basis_vectors, Lx=Lx, Ly=Ly, pbc=pbc)

    dist_matrix = np.full((n_sites, n_sites), np.inf, dtype=float)
    for t in translations:
        dist_matrix = np.minimum(dist_matrix, np.linalg.norm(disp + t, axis=-1))

    offdiag = ~np.eye(n_sites, dtype=bool)
    dist_shell = np.round(dist_matrix, 12)
    r_values = np.unique(dist_shell[offdiag])

    g_r = np.empty(r_values.size, dtype=float)
    norm_density = (n_fermions / n_sites) ** 2
    for i, r in enumerate(r_values):
        mask = (dist_shell == r) & offdiag
        g_r[i] = corr_matrix[mask].mean().real / norm_density

    if r_values.size == 0 or not np.isclose(r_values[0], 0.0):
        r_values_plot = np.concatenate(([0.0], r_values))
        g_r_plot = np.concatenate(([0.0], g_r))
    else:
        r_values_plot = r_values.copy()
        g_r_plot = g_r.copy()
        g_r_plot[0] = 0.0

    return {
        "charge_density": charge_density,
        "corr_matrix": corr_matrix,
        "dist_matrix": dist_matrix,
        "translations": translations,
        "r_values_plot": r_values_plot,
        "g_r_plot": g_r_plot,
    }


def map_radial_g_to_minimum_image_2d(
    positions: np.ndarray,
    basis_coords: np.ndarray,
    translations: np.ndarray,
    r_values_plot: np.ndarray,
    g_r_plot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # Written with Codex 02-18-26.
    n_sites = int(positions.shape[0])
    coords = np.asarray(basis_coords)

    a_sublattice = np.where(coords[:, 2] == 0)[0]
    a_origin = np.where((coords[:, 0] == 0) & (coords[:, 1] == 0) & (coords[:, 2] == 0))[0]
    if a_origin.size > 0:
        origin_idx = int(a_origin[0])
    elif a_sublattice.size > 0:
        origin_idx = int(a_sublattice[0])
    else:
        raise ValueError("Could not find an A-sublattice site for the g(r) origin.")

    origin_pos = positions[origin_idx]
    rel = positions - origin_pos[np.newaxis, :]

    rel_min = np.empty_like(rel)
    r_site = np.empty(n_sites, dtype=float)
    for i in range(n_sites):
        best_norm2 = np.inf
        best_vec = None
        for t in translations:
            vec = rel[i] + t
            norm2 = float(np.dot(vec, vec))
            if norm2 < best_norm2:
                best_norm2 = norm2
                best_vec = vec

        if best_vec is None:
            raise RuntimeError("Failed to find minimum-image representative.")
        rel_min[i] = best_vec
        r_site[i] = np.sqrt(best_norm2)

    g_lookup = {
        float(np.round(r, 12)): float(g)
        for r, g in zip(r_values_plot, g_r_plot)
    }
    r_site_shell = np.round(r_site, 12)
    g_site = np.array([g_lookup[float(r)] for r in r_site_shell], dtype=float)
    return rel_min, g_site, origin_idx, r_site


def static_structure_factor(
    samples: np.ndarray,
    positions: np.ndarray,
    basis_vectors: np.ndarray,
    Lx: int,
    Ly: int,
    n_fermions: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    b1, b2 = reciprocal_from_basis(basis_vectors)

    q_list = np.empty((Lx * Ly, 2), dtype=float)
    idx = 0
    for n1 in range(Lx):
        for n2 in range(Ly):
            q_raw = (n1 / Lx) * b1 + (n2 / Ly) * b2
            q_list[idx] = fold_to_shortest_k(q_raw, b1, b2)
            idx += 1

    phases = np.exp(-1j * (positions @ q_list.T))
    rho_q = samples @ phases
    s_q = np.mean(np.abs(rho_q) ** 2, axis=0) / n_fermions
    return q_list, s_q


def radial_average_structure_factor(
    q_list: np.ndarray,
    s_q: np.ndarray,
    set_q0_to_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    q_abs = np.linalg.norm(q_list, axis=1)

    s_q_plot = s_q.copy()
    if set_q0_to_zero:
        s_q_plot[q_abs < 1e-12] = 0.0

    q_shell = np.round(q_abs, 12)
    q_shell_unique = np.unique(q_shell)

    s_q_abs = np.empty_like(q_shell_unique)
    s_q_abs_err = np.empty_like(q_shell_unique)
    for i, qv in enumerate(q_shell_unique):
        vals = s_q_plot.real[q_shell == qv]
        s_q_abs[i] = vals.mean()
        s_q_abs_err[i] = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0

    return q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot
