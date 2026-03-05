from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType
from .attention import SpatialFactoredMultiHeadAttention


class SpatialEncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    displacement_only_attention: bool = True
    mlp_hidden_factor: int = 4
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6

    def setup(self):
        # Written with Codex 02-18-26.
        self.attn = SpatialFactoredMultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            pair_classes=self.pair_classes,
            pair_distances=self.pair_distances,
            displacement_only_attention=self.displacement_only_attention,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            xi_epsilon=self.xi_epsilon,
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
        # Written with Codex 02-18-26.
        y = x + self.attn(self.layer_norm_1(x))
        z = self.ff_dense_2(nn.gelu(self.ff_dense_1(self.layer_norm_2(y))))
        return y + z


class SpatialEncoder(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    displacement_only_attention: bool = True
    mlp_hidden_factor: int = 4
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6

    def setup(self):
        # Written with Codex 02-18-26.
        self.layers = [
            SpatialEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                pair_classes=self.pair_classes,
                pair_distances=self.pair_distances,
                displacement_only_attention=self.displacement_only_attention,
                mlp_hidden_factor=self.mlp_hidden_factor,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                xi_epsilon=self.xi_epsilon,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        for layer in self.layers:
            x = layer(x)
        return x
