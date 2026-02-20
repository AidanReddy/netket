from __future__ import annotations

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from ._common import DType, default_kernel_init


def _logdet_cmplx(matrix: jax.Array) -> jax.Array:
    # Written with Codex 02-18-26.
    sign, logabsdet = jnp.linalg.slogdet(matrix)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


def _constant_array_initializer(array: Any):
    # Written with Codex 02-19-26.
    array = jnp.asarray(array)

    def init(key: jax.Array, shape: tuple[int, ...], dtype: DType = jnp.float32):
        # Written with Codex 02-19-26.
        del key
        if tuple(shape) != tuple(array.shape):
            raise ValueError(
                f"Initializer shape mismatch: got shape={shape}, expected {array.shape}."
            )
        return jnp.asarray(array, dtype=dtype)

    return init


def _hashable_matrix_literal(array_like: Any):
    # Written with Codex 02-19-26.
    if array_like is None:
        return None
    arr = np.asarray(array_like)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a rank-2 matrix literal, got shape={arr.shape}."
        )
    return tuple(tuple(v for v in row) for row in arr.tolist())


class LogSlaterDeterminant(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    kernel_init: Any = default_kernel_init
    param_dtype: DType = complex
    split_complex_params: bool = False
    initial_m_orbitals: Any | None = None

    def __post_init__(self):
        # Written with Codex 02-19-26.
        object.__setattr__(
            self,
            "initial_m_orbitals",
            _hashable_matrix_literal(self.initial_m_orbitals),
        )
        super().__post_init__()

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        m_shape = (self.hilbert.n_orbitals, self.hilbert.n_fermions)
        m0 = None
        if self.initial_m_orbitals is not None:
            m0 = jnp.asarray(self.initial_m_orbitals)
            if tuple(m0.shape) != m_shape:
                raise ValueError(
                    "initial_m_orbitals has invalid shape. "
                    f"Expected {m_shape}, got {tuple(m0.shape)}."
                )
        if self.split_complex_params:
            if jnp.issubdtype(jnp.dtype(self.param_dtype), jnp.complexfloating):
                raise ValueError(
                    "split_complex_params=True requires a real param_dtype."
                )
            m_real_init = (
                self.kernel_init
                if m0 is None
                else _constant_array_initializer(jnp.real(m0))
            )
            m_imag_init = (
                self.kernel_init
                if m0 is None
                else _constant_array_initializer(jnp.imag(m0))
            )
            m_real = self.param(
                "M_re",
                m_real_init,
                m_shape,
                self.param_dtype,
            )
            m_imag = self.param(
                "M_im",
                m_imag_init,
                m_shape,
                self.param_dtype,
            )
            m_orbitals = jax.lax.complex(m_real, m_imag)
        else:
            m_init = self.kernel_init if m0 is None else _constant_array_initializer(m0)
            m_orbitals = self.param(
                "M",
                m_init,
                m_shape,
                self.param_dtype,
            )

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n_single: jax.Array) -> jax.Array:
            # Written with Codex 02-18-26.
            occupied = n_single.nonzero(size=self.hilbert.n_fermions)[0]
            slater_matrix = m_orbitals[occupied]
            return _logdet_cmplx(slater_matrix)

        return log_sd(n)


class LogSlaterJastrow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    kernel_init: Any = default_kernel_init
    param_dtype: DType = complex
    jastrow_param_dtype: DType = complex

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        log_slater = LogSlaterDeterminant(
            self.hilbert,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            name="slater",
        )(n)
        log_jastrow = nk.models.Jastrow(
            param_dtype=self.jastrow_param_dtype,
            name="jastrow",
        )(n)
        return log_slater + log_jastrow
