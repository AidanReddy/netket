from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType, log_cosh


class ComplexLogCoshOutputHead(nn.Module):
    d_model: int
    hidden_dim: int | None = None
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        # Written with Codex 02-18-26.
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.d_model

        self.out_layer_norm = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="out_ln",
        )
        self.real_ln = nn.LayerNorm(
            use_scale=True,
            use_bias=True,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="real_ln",
        )
        self.imag_ln = nn.LayerNorm(
            use_scale=True,
            use_bias=True,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="imag_ln",
        )
        self.real_dense = nn.Dense(
            hidden_dim,
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="real_dense",
        )
        self.imag_dense = nn.Dense(
            hidden_dim,
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="imag_dense",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        z = self.out_layer_norm(jnp.sum(x, axis=1))
        real_features = self.real_ln(self.real_dense(z))
        imag_features = self.imag_ln(self.imag_dense(z))
        out = real_features + 1j * imag_features
        return jnp.sum(log_cosh(out), axis=-1)
