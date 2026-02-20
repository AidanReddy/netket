from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from ._common import DType, default_kernel_init
from .slater import LogSlaterDeterminant, _hashable_matrix_literal
from .vit import LogSpatialViT


class LogSlaterSpatialViT(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    num_layers: int
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    slater_kernel_init: Any = default_kernel_init
    slater_param_dtype: DType = jnp.float64
    slater_split_complex_params: bool = True
    slater_initial_m_orbitals: Any | None = None
    vit_param_dtype: DType = jnp.float64
    mlp_hidden_factor: int = 4
    output_hidden_dim: int | None = None
    vit_kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6

    def __post_init__(self):
        # Written with Codex 02-19-26.
        object.__setattr__(
            self,
            "slater_initial_m_orbitals",
            _hashable_matrix_literal(self.slater_initial_m_orbitals),
        )
        super().__post_init__()

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        log_slater = LogSlaterDeterminant(
            hilbert=self.hilbert,
            kernel_init=self.slater_kernel_init,
            param_dtype=self.slater_param_dtype,
            split_complex_params=self.slater_split_complex_params,
            initial_m_orbitals=self.slater_initial_m_orbitals,
            name="slater",
        )(n)
        log_vit = LogSpatialViT(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            pair_classes=self.pair_classes,
            pair_distances=self.pair_distances,
            mlp_hidden_factor=self.mlp_hidden_factor,
            output_hidden_dim=self.output_hidden_dim,
            param_dtype=self.vit_param_dtype,
            kernel_init=self.vit_kernel_init,
            xi_epsilon=self.xi_epsilon,
            name="vit",
        )(n)
        return log_slater + log_vit
