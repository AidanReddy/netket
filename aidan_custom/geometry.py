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


def high_symmetry_points(b1: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, ...]:
    # Written with Codex 02-18-26.
    gamma = np.array([0.0, 0.0])
    k_raw = (2.0 * b1 + b2) / 3.0
    kp_raw = (b1 + 2.0 * b2) / 3.0
    k = fold_to_shortest_k(k_raw, b1, b2)
    kp = fold_to_shortest_k(kp_raw, b1, b2)
    mpt = 0.5 * (k + kp)
    return gamma, k, mpt, kp


def first_bz_hexagon_vertices(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    # Written with Codex 02-18-26.
    return np.array(
        [
            fold_to_shortest_k((2.0 * b1 + b2) / 3.0, b1, b2),
            fold_to_shortest_k((b1 + 2.0 * b2) / 3.0, b1, b2),
            fold_to_shortest_k((-b1 + b2) / 3.0, b1, b2),
            fold_to_shortest_k(-(2.0 * b1 + b2) / 3.0, b1, b2),
            fold_to_shortest_k(-(b1 + 2.0 * b2) / 3.0, b1, b2),
            fold_to_shortest_k((b1 - b2) / 3.0, b1, b2),
        ]
    )


def k_path_gamma_k_m_kp_gamma(
    n_points_per_segment: int,
    b1: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # Written with Codex 02-18-26.
    gamma, k, mpt, kp = high_symmetry_points(b1, b2)
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
