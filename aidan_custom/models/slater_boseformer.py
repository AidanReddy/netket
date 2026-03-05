from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from ._common import DType, default_kernel_init
from .boseformer import LogBoseFormerProduct
from .slater import LogSlaterDeterminant, _hashable_matrix_literal


class LogSlaterBoseFormer(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    positions: Any
    g_vectors: Any
    num_layers: int
    d_model: int
    n_heads: int
    slater_kernel_init: Any = default_kernel_init
    slater_param_dtype: DType = jnp.float64
    slater_split_complex_params: bool = True
    slater_initial_m_orbitals: Any | None = None
    embedding_type: str = "periodic"
    boseformer_param_dtype: DType = jnp.float64
    mlp_hidden_factor: int = 4
    boseformer_kernel_init: Any = nn.initializers.xavier_uniform()
    boseformer_orbital_kernel_init: Any = jax.nn.initializers.zeros

    def __post_init__(self):
        # Written with Codex 02-19-26.
        object.__setattr__(
            self,
            "slater_initial_m_orbitals",
            _hashable_matrix_literal(self.slater_initial_m_orbitals),
        )
        object.__setattr__(self, "positions", _hashable_matrix_literal(self.positions))
        object.__setattr__(self, "g_vectors", _hashable_matrix_literal(self.g_vectors))
        super().__post_init__()

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-19-26.
        log_slater = LogSlaterDeterminant(
            hilbert=self.hilbert,
            kernel_init=self.slater_kernel_init,
            param_dtype=self.slater_param_dtype,
            split_complex_params=self.slater_split_complex_params,
            initial_m_orbitals=self.slater_initial_m_orbitals,
            name="slater",
        )(n)
        log_boseformer = LogBoseFormerProduct(
            n_particles=self.hilbert.n_fermions,
            positions=self.positions,
            g_vectors=self.g_vectors,
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_hidden_factor=self.mlp_hidden_factor,
            embedding_type=self.embedding_type,
            param_dtype=self.boseformer_param_dtype,
            kernel_init=self.boseformer_kernel_init,
            orbital_kernel_init=self.boseformer_orbital_kernel_init,
            name="boseformer",
        )(n)
        return log_slater + log_boseformer
