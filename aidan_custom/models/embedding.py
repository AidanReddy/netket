from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType
from .slater import _hashable_matrix_literal


class SiteOccupancyEmbedding(nn.Module):
    d_model: int
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        x = jnp.atleast_2d(n)
        x = x.astype(self.param_dtype)
        # Map occupancies from {0, 1} to {-1, +1} before embedding.
        x = jnp.where(x > 0, 1.0, -1.0).astype(self.param_dtype)
        x = x[..., None]
        x = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="embed",
        )(x)
        return x


class OccupiedPeriodicFeatureEmbedding(nn.Module):
    n_particles: int
    positions: Any
    g_vectors: Any
    d_model: int
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def __post_init__(self):
        # Written with Codex 02-19-26.
        object.__setattr__(self, "positions", _hashable_matrix_literal(self.positions))
        object.__setattr__(self, "g_vectors", _hashable_matrix_literal(self.g_vectors))
        super().__post_init__()

    def setup(self):
        # Written with Codex 02-19-26.
        positions = jnp.asarray(self.positions, dtype=jnp.float64)
        g_vectors = jnp.asarray(self.g_vectors, dtype=jnp.float64)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (n_sites, 2).")
        if g_vectors.shape != (2, 2):
            raise ValueError("g_vectors must have shape (2, 2).")
        if int(self.n_particles) <= 0:
            raise ValueError("n_particles must be positive.")
        if int(self.n_particles) > int(positions.shape[0]):
            raise ValueError("n_particles cannot exceed number of sites.")

        phase = positions @ g_vectors.T
        site_features = jnp.stack(
            (
                jnp.cos(phase[:, 0]),
                jnp.sin(phase[:, 0]),
                jnp.cos(phase[:, 1]),
                jnp.sin(phase[:, 1]),
            ),
            axis=-1,
        )
        self._site_features = site_features.astype(self.param_dtype)
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="embed",
        )

    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        x = jnp.atleast_2d(n)
        if x.shape[-1] != self._site_features.shape[0]:
            raise ValueError("Input occupation dimension does not match positions.")

        # Deterministic token order: occupied sites are listed by ascending index.
        occupied_indices = jax.vmap(
            lambda n_single: jnp.nonzero(
                n_single > 0,
                size=self.n_particles,
                fill_value=0,
            )[0]
        )(x)
        particle_features = jnp.take(self._site_features, occupied_indices, axis=0)
        return self.embed(particle_features)
