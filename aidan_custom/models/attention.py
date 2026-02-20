from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ._common import DType


class SpatialFactoredMultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6
    xi_init: float = 3.0

    def setup(self):
        # Written with Codex 02-18-26.
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        pair_classes_arr = jnp.asarray(self.pair_classes, dtype=jnp.int32)
        pair_distances_arr = jnp.asarray(self.pair_distances, dtype=jnp.float64).reshape(-1)

        if pair_classes_arr.ndim != 2 or pair_classes_arr.shape[0] != pair_classes_arr.shape[1]:
            raise ValueError("pair_classes must have shape (n_sites, n_sites).")
        if pair_distances_arr.ndim != 1 or pair_distances_arr.size == 0:
            raise ValueError("pair_distances must be a non-empty 1D array.")
        n_pair_classes = pair_distances_arr.shape[0]

        self._pair_classes = pair_classes_arr
        self._pair_distances = pair_distances_arr.astype(self.param_dtype)
        one_hot_classes = jax.nn.one_hot(
            pair_classes_arr,
            n_pair_classes,
            dtype=self.param_dtype,
        )
        class_counts_by_row = jnp.sum(one_hot_classes, axis=1)
        self._class_counts_by_row = class_counts_by_row
        first_row_counts = class_counts_by_row[0]
        self._class_counts = first_row_counts

        self.value_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="value_proj",
        )
        self.output_proj = nn.Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="output_proj",
        )

        self.alpha = self.param(
            "alpha",
            self.kernel_init,
            (self.n_heads, n_pair_classes),
            self.param_dtype,
        )
        self.raw_xi = self.param(
            "raw_xi",
            nn.initializers.constant(
                jnp.log(
                    jnp.expm1(
                        jnp.asarray(
                            max(self.xi_init - self.xi_epsilon, 1.0e-12),
                            dtype=self.param_dtype,
                        )
                    )
                )
            ),
            (self.n_heads,),
            self.param_dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        if x.ndim != 3:
            raise ValueError("Attention input must have shape (batch, n_sites, d_model).")
        if x.shape[-1] != self.d_model:
            raise ValueError("Last dimension of attention input must equal d_model.")
        if x.shape[1] != self._pair_classes.shape[0]:
            raise ValueError("n_sites mismatch between input and pair_classes.")

        batch_size, n_sites, _ = x.shape
        d_head = self.d_model // self.n_heads

        v = self.value_proj(x)
        v = v.reshape(batch_size, n_sites, self.n_heads, d_head)
        v = jnp.transpose(v, (0, 2, 1, 3))

        xi = jax.nn.softplus(self.raw_xi) + self.xi_epsilon
        envelope = jnp.exp(-self._pair_distances[None, :] / xi[:, None])
        is_uniform_row_class_count = jnp.all(
            self._class_counts_by_row == self._class_counts[jnp.newaxis, :]
        )

        def _kernel_pairs_uniform(args):
            # Written with Codex 02-18-26.
            envelope, alpha = args
            denom = jnp.sum(
                envelope * self._class_counts[jnp.newaxis, :],
                axis=-1,
                keepdims=True,
            )
            envelope_norm_by_class = envelope / denom
            return (
                alpha[:, self._pair_classes]
                * envelope_norm_by_class[:, self._pair_classes]
            )

        def _kernel_pairs_general(args):
            # Written with Codex 02-18-26.
            envelope, alpha = args
            denom = jnp.einsum(
                "hc,ic->hi",
                envelope,
                self._class_counts_by_row,
                optimize=True,
            )
            envelope_pairs = envelope[:, self._pair_classes]
            envelope_norm = envelope_pairs / denom[:, :, None]
            return alpha[:, self._pair_classes] * envelope_norm

        kernel_pairs = jax.lax.cond(
            is_uniform_row_class_count,
            _kernel_pairs_uniform,
            _kernel_pairs_general,
            operand=(envelope, self.alpha),
        )

        attended = jnp.einsum("hij,bhjd->bhid", kernel_pairs, v, optimize=True)
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, n_sites, self.d_model)
        return self.output_proj(attended)
