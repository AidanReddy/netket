from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType
from .embedding import OccupiedPeriodicFeatureEmbedding
from .slater import _hashable_matrix_literal


class ParticleQKVSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        # Written with Codex 02-19-26.
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self._d_head = self.d_model // self.n_heads
        self._scale = 1.0 / math.sqrt(self._d_head)

        self.q_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="q_proj",
        )
        self.k_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="k_proj",
        )
        self.v_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="v_proj",
        )
        self.out_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="out_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        if x.ndim != 3:
            raise ValueError("Attention input must have shape (batch, n_particles, d_model).")
        if x.shape[-1] != self.d_model:
            raise ValueError("Last dimension of attention input must equal d_model.")

        batch_size, n_particles, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, n_particles, self.n_heads, self._d_head)
        k = self.k_proj(x).reshape(batch_size, n_particles, self.n_heads, self._d_head)
        v = self.v_proj(x).reshape(batch_size, n_particles, self.n_heads, self._d_head)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        logits = jnp.einsum("bhid,bhjd->bhij", q, k, optimize=True) * self._scale
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhij,bhjd->bhid", weights, v, optimize=True)

        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
            batch_size, n_particles, self.d_model
        )
        return self.out_proj(attended)


class BoseFormerEncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_hidden_factor: int = 4
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        # Written with Codex 02-19-26.
        self.attn = ParticleQKVSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="attn",
        )
        self.layer_norm_1 = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="ln1",
        )
        self.layer_norm_2 = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="ln2",
        )
        self.ff_dense_1 = nn.Dense(
            self.mlp_hidden_factor * self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="ff_dense_1",
        )
        self.ff_dense_2 = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="ff_dense_2",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        y = x + self.attn(self.layer_norm_1(x))
        z = self.ff_dense_2(nn.gelu(self.ff_dense_1(self.layer_norm_2(y))))
        return y + z


class BoseFormerEncoder(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    mlp_hidden_factor: int = 4
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        # Written with Codex 02-19-26.
        self.layers = [
            BoseFormerEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_hidden_factor=self.mlp_hidden_factor,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        for layer in self.layers:
            x = layer(x)
        return x


class ComplexParticleOrbitalProductHead(nn.Module):
    d_model: int
    param_dtype: DType = jnp.float64
    kernel_init: Any = jax.nn.initializers.zeros

    def setup(self):
        # Written with Codex 02-19-26.
        self.pre_head_ln = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="pre_head_ln",
        )
        self.real_dense = nn.Dense(
            1,
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="real_dense",
        )
        self.imag_dense = nn.Dense(
            1,
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="imag_dense",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        z = self.pre_head_ln(x)
        real_log_orbitals = self.real_dense(z)[..., 0]
        imag_log_orbitals = self.imag_dense(z)[..., 0]
        return jnp.sum(real_log_orbitals + 1j * imag_log_orbitals, axis=1)


class LogBoseFormerProduct(nn.Module):
    n_particles: int
    positions: Any
    g_vectors: Any
    num_layers: int
    d_model: int
    n_heads: int
    mlp_hidden_factor: int = 4
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    orbital_kernel_init: Any = jax.nn.initializers.zeros

    def __post_init__(self):
        # Written with Codex 02-19-26.
        object.__setattr__(self, "positions", _hashable_matrix_literal(self.positions))
        object.__setattr__(self, "g_vectors", _hashable_matrix_literal(self.g_vectors))
        super().__post_init__()

    def setup(self):
        # Written with Codex 02-19-26.
        self.embedding = OccupiedPeriodicFeatureEmbedding(
            n_particles=self.n_particles,
            positions=self.positions,
            g_vectors=self.g_vectors,
            d_model=self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="embed",
        )
        self.encoder = BoseFormerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_hidden_factor=self.mlp_hidden_factor,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="encoder",
        )
        self.output_head = ComplexParticleOrbitalProductHead(
            d_model=self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=self.orbital_kernel_init,
            name="output_head",
        )

    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        x = self.embedding(n)
        y = self.encoder(x)
        return self.output_head(y)
