from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import netket as nk


def _as_2d_periodic_boundary_flags(pbc: Any) -> jax.Array:
    # Written with Codex 02-18-26.
    pbc = jnp.asarray(pbc, dtype=bool).reshape(-1)
    if pbc.size == 1:
        pbc = jnp.repeat(pbc, 2)
    if pbc.size != 2:
        raise ValueError("pbc must be a scalar bool or have exactly two entries.")
    return pbc


def _minimum_image_translations(
    basis_vectors: Any,
    Lx: int,
    Ly: int,
    pbc: Any,
) -> jax.Array:
    # Written with Codex 02-18-26.
    basis_vectors = jnp.asarray(basis_vectors, dtype=jnp.float64)
    pbc = _as_2d_periodic_boundary_flags(pbc)
    shift_n1 = (-1, 0, 1) if bool(pbc[0]) else (0,)
    shift_n2 = (-1, 0, 1) if bool(pbc[1]) else (0,)
    translations = [
        (n1 * Lx) * basis_vectors[0] + (n2 * Ly) * basis_vectors[1]
        for n1 in shift_n1
        for n2 in shift_n2
    ]
    return jnp.stack(translations, axis=0)


def make_translation_equivariant_pair_data(
    positions: Any,
    basis_vectors: Any,
    Lx: int,
    Ly: int,
    pbc: Any,
    round_decimals: int = 12,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Written with Codex 02-18-26.
    positions = jnp.asarray(positions, dtype=jnp.float64)
    basis_vectors = jnp.asarray(basis_vectors, dtype=jnp.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_sites, 2).")
    if basis_vectors.shape != (2, 2):
        raise ValueError("basis_vectors must have shape (2, 2).")

    translations = _minimum_image_translations(
        basis_vectors=basis_vectors,
        Lx=Lx,
        Ly=Ly,
        pbc=pbc,
    )

    # Spatial displacement for alpha_(i-j): displacement from site j to site i.
    rel = positions[:, jnp.newaxis, :] - positions[jnp.newaxis, :, :]
    candidates = rel[jnp.newaxis, :, :, :] + translations[:, jnp.newaxis, jnp.newaxis, :]
    norms2 = jnp.sum(candidates * candidates, axis=-1)
    norms2_key = jnp.round(norms2, decimals=round_decimals)
    min_norm2_key = jnp.min(norms2_key, axis=0, keepdims=True)
    # Tie-breaking at minimum-image boundaries: choose the earliest translation
    # in the deterministic (-1,0,1) x (-1,0,1) ordering.
    is_min = norms2_key == min_norm2_key
    rank = jnp.arange(translations.shape[0], dtype=jnp.int32)[:, jnp.newaxis, jnp.newaxis]
    masked_rank = jnp.where(is_min, rank, jnp.iinfo(jnp.int32).max)
    best_idx = jnp.argmin(masked_rank, axis=0, keepdims=True)
    rel_min = jnp.take_along_axis(
        candidates,
        best_idx[:, :, :, jnp.newaxis],
        axis=0,
    )[0]

    rel_keys = jnp.round(rel_min, decimals=round_decimals)
    unique_vectors, inverse = jnp.unique(
        rel_keys.reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )

    pair_classes = inverse.reshape(rel_min.shape[:2]).astype(jnp.int32)
    class_distances = jnp.linalg.norm(unique_vectors, axis=1).astype(jnp.float64)
    class_vectors = unique_vectors.astype(jnp.float64)
    return pair_classes, class_distances, class_vectors


def make_translation_equivariant_pair_data_from_graph(
    graph: nk.graph.AbstractGraph,
    round_decimals: int = 12,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Written with Codex 02-18-26.
    extent = jnp.asarray(graph.extent, dtype=jnp.int32).reshape(-1)
    if extent.size != 2:
        raise ValueError("Only 2D graphs with extent=[Lx, Ly] are currently supported.")
    return make_translation_equivariant_pair_data(
        positions=jnp.asarray(graph.positions, dtype=jnp.float64),
        basis_vectors=jnp.asarray(graph.basis_vectors, dtype=jnp.float64),
        Lx=int(extent[0]),
        Ly=int(extent[1]),
        pbc=jnp.asarray(graph.pbc, dtype=bool),
        round_decimals=round_decimals,
    )


def make_supercell_reciprocal_vectors(
    basis_vectors: Any,
    Lx: int,
    Ly: int,
) -> jax.Array:
    # Written with Codex 02-19-26.
    basis_vectors = jnp.asarray(basis_vectors, dtype=jnp.float64)
    if basis_vectors.shape != (2, 2):
        raise ValueError("basis_vectors must have shape (2, 2).")
    if int(Lx) <= 0 or int(Ly) <= 0:
        raise ValueError("Lx and Ly must be positive integers.")

    # `basis_vectors` stores primitive vectors as rows: [a1; a2].
    # Reciprocal rows [b1; b2] satisfy b_mu · a_nu = 2π δ_{mu,nu}.
    reciprocal = (2.0 * jnp.pi) * jnp.linalg.inv(basis_vectors.T)
    supercell_scale = jnp.asarray([Lx, Ly], dtype=jnp.float64)[:, jnp.newaxis]
    return reciprocal / supercell_scale


def make_supercell_reciprocal_vectors_from_graph(
    graph: nk.graph.AbstractGraph,
) -> jax.Array:
    # Written with Codex 02-19-26.
    extent = jnp.asarray(graph.extent, dtype=jnp.int32).reshape(-1)
    if extent.size != 2:
        raise ValueError("Only 2D graphs with extent=[Lx, Ly] are currently supported.")
    return make_supercell_reciprocal_vectors(
        basis_vectors=jnp.asarray(graph.basis_vectors, dtype=jnp.float64),
        Lx=int(extent[0]),
        Ly=int(extent[1]),
    )
