from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType
from .embedding import SiteOccupancyEmbedding
from .encoder import SpatialEncoder
from .output_head import ComplexLogCoshOutputHead


class LogSpatialViT(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    displacement_only_attention: bool = True
    mlp_hidden_factor: int = 4
    output_hidden_dim: int | None = None
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        x = SiteOccupancyEmbedding(
            d_model=self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="embed",
        )(n)
        y = SpatialEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            pair_classes=self.pair_classes,
            pair_distances=self.pair_distances,
            displacement_only_attention=self.displacement_only_attention,
            mlp_hidden_factor=self.mlp_hidden_factor,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            xi_epsilon=self.xi_epsilon,
            name="encoder",
        )(x)
        return ComplexLogCoshOutputHead(
            d_model=self.d_model,
            hidden_dim=self.output_hidden_dim,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="output_head",
        )(y)
