from __future__ import annotations

import numpy as np

PRIMITIVE_A1 = np.array([1.0, 0.0])
PRIMITIVE_A2 = np.array([0.5, np.sqrt(3.0) / 2.0])
SUBLATTICE_A_OFFSET = np.array([0.5, 0.5 / np.sqrt(3.0)])
SUBLATTICE_B_OFFSET = np.array([1.0, 1.0 / np.sqrt(3.0)])

DELTA_VECTORS = (
    SUBLATTICE_B_OFFSET - SUBLATTICE_A_OFFSET,
    SUBLATTICE_B_OFFSET - SUBLATTICE_A_OFFSET - PRIMITIVE_A1,
    SUBLATTICE_B_OFFSET - SUBLATTICE_A_OFFSET - PRIMITIVE_A2,
)

ANNN_VECTORS = (
    DELTA_VECTORS[1] - DELTA_VECTORS[2],
    DELTA_VECTORS[2] - DELTA_VECTORS[0],
    DELTA_VECTORS[0] - DELTA_VECTORS[1],
)


def reciprocal_vectors(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    area = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = (2.0 * np.pi / area) * np.array([a2[1], -a2[0]])
    b2 = (2.0 * np.pi / area) * np.array([-a1[1], a1[0]])
    return b1, b2


def reciprocal_from_basis(basis_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    return reciprocal_vectors(basis_vectors[0], basis_vectors[1])


def fold_to_shortest_k(
    kvec: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    search_radius: int = 2,
) -> np.ndarray:
    # Written with Codex 02-18-26.
    basis = np.column_stack((b1, b2))
    coeffs = np.linalg.solve(basis, kvec)
    n0 = np.rint(coeffs).astype(int)

    best_k = None
    best_norm2 = np.inf
    for n1 in range(n0[0] - search_radius, n0[0] + search_radius + 1):
        for n2 in range(n0[1] - search_radius, n0[1] + search_radius + 1):
            krev = kvec - n1 * b1 - n2 * b2
            norm2 = float(np.dot(krev, krev))
            if norm2 < best_norm2:
                best_norm2 = norm2
                best_k = krev

    if best_k is None:
        raise RuntimeError("Failed to fold k-point to shortest representative.")
    return best_k


def _first_shell_reciprocal_vectors(
    b1: np.ndarray,
    b2: np.ndarray,
    search_radius: int = 2,
    shell_tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    g_vectors = []
    for n1 in range(-search_radius, search_radius + 1):
        for n2 in range(-search_radius, search_radius + 1):
            if n1 == 0 and n2 == 0:
                continue
            g_vectors.append(n1 * b1 + n2 * b2)

    g_all = np.asarray(g_vectors, dtype=float)
    g_norm2 = np.einsum("ij,ij->i", g_all, g_all)
    min_norm2 = float(np.min(g_norm2))
    shell_mask = g_norm2 <= (1.0 + shell_tolerance) * min_norm2
    g_shell = g_all[shell_mask]
    if g_shell.shape[0] < 6:
        raise RuntimeError(
            "Failed to identify the first reciprocal shell; expected at least six vectors."
        )
    return g_all, g_norm2, g_shell


def _hexagon_vertices_from_reciprocal_basis(
    b1: np.ndarray,
    b2: np.ndarray,
    search_radius: int = 2,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    g_all, g_norm2, g_shell = _first_shell_reciprocal_vectors(
        b1=b1,
        b2=b2,
        search_radius=search_radius,
    )
    half_norm2 = 0.5 * g_norm2

    vertices = []
    n_shell = g_shell.shape[0]
    for i in range(n_shell - 1):
        gi = g_shell[i]
        rhs_i = 0.5 * float(np.dot(gi, gi))
        for j in range(i + 1, n_shell):
            gj = g_shell[j]
            det = gi[0] * gj[1] - gi[1] * gj[0]
            if abs(det) <= tolerance:
                continue
            rhs_j = 0.5 * float(np.dot(gj, gj))
            mat = np.array([[gi[0], gi[1]], [gj[0], gj[1]]], dtype=float)
            rhs = np.array([rhs_i, rhs_j], dtype=float)
            kval = np.linalg.solve(mat, rhs)
            if np.all((g_all @ kval) <= (half_norm2 + tolerance)):
                vertices.append(kval)

    if len(vertices) == 0:
        raise RuntimeError("Failed to construct first-BZ vertices from reciprocal basis.")

    verts = np.asarray(vertices, dtype=float)
    keys = np.round(verts, decimals=12)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    verts = verts[np.sort(unique_idx)]

    if verts.shape[0] < 6:
        raise RuntimeError(
            f"Expected at least six unique first-BZ vertices, got {verts.shape[0]}."
        )
    if verts.shape[0] > 6:
        radii2 = np.einsum("ij,ij->i", verts, verts)
        target_r2 = float(np.max(radii2))
        keep = radii2 >= (1.0 - 1.0e-9) * target_r2
        verts = verts[keep]

    center = np.mean(verts, axis=0)
    angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
    return verts[np.argsort(angles)]


def high_symmetry_points(
    b1: np.ndarray,
    b2: np.ndarray,
    fold: bool = True,
) -> tuple[np.ndarray, ...]:
    # Written with Codex 02-20-26.
    gamma = np.array([0.0, 0.0])
    if fold:
        verts = _hexagon_vertices_from_reciprocal_basis(b1=b1, b2=b2)
        mids = 0.5 * (verts + np.roll(verts, -1, axis=0))
        ref_dir = b1 + b2
        ref_norm = float(np.linalg.norm(ref_dir))
        if ref_norm <= 1.0e-14:
            ref_dir = b1
            ref_norm = float(np.linalg.norm(ref_dir))
        if ref_norm <= 1.0e-14:
            raise ValueError("Reciprocal basis vectors must be linearly independent.")
        scores = mids @ (ref_dir / ref_norm)
        idx = int(np.argmax(scores))
        k = verts[idx]
        kp = verts[(idx + 1) % verts.shape[0]]
        mpt = mids[idx]
    else:
        # Use an effective reciprocal basis with obtuse mutual angle.
        # The closed-form K/K' expressions assume this convention.
        if float(np.dot(b1, b2)) > 0.0:
            b1_eff = b1
            b2_eff = -b2
        else:
            b1_eff = b1
            b2_eff = b2

        k = (2.0 * b1_eff + b2_eff) / 3.0
        kp = (b1_eff + 2.0 * b2_eff) / 3.0
        mpt = 0.5 * (k + kp)
    return gamma, k, mpt, kp


def first_bz_hexagon_vertices(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    # Written with Codex 02-20-26.
    return _hexagon_vertices_from_reciprocal_basis(b1=b1, b2=b2)


def k_path_gamma_k_m_kp_gamma(
    n_points_per_segment: int,
    b1: np.ndarray,
    b2: np.ndarray,
    fold: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # Written with Codex 02-20-26.
    gamma, k, mpt, kp = high_symmetry_points(b1, b2, fold=fold)
    nodes = [gamma, k, mpt, kp, gamma]
    labels = [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"]

    k_points = [nodes[0]]
    x = [0.0]
    x_nodes = [0.0]

    for i in range(len(nodes) - 1):
        p0 = nodes[i]
        p1 = nodes[i + 1]
        for j in range(1, n_points_per_segment + 1):
            alpha = j / n_points_per_segment
            p = (1.0 - alpha) * p0 + alpha * p1
            ds = np.linalg.norm(p - k_points[-1])
            k_points.append(p)
            x.append(x[-1] + ds)
        x_nodes.append(x[-1])

    return np.array(k_points), np.array(x), np.array(x_nodes), labels


def sample_shortest_representative_bz(
    nx: int,
    ny: int,
    b1: np.ndarray,
    b2: np.ndarray,
    unique: bool = True,
) -> np.ndarray:
    # Written with Codex 02-18-26.
    kpts = np.empty((nx * ny, 2), dtype=float)
    idx = 0
    for n1 in range(nx):
        for n2 in range(ny):
            kvec = (n1 / nx) * b1 + (n2 / ny) * b2
            kpts[idx] = fold_to_shortest_k(kvec, b1, b2)
            idx += 1

    if not unique:
        return kpts

    keys = np.round(kpts, decimals=12)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return kpts[np.sort(unique_idx)]
