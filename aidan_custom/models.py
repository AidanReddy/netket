from __future__ import annotations

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
from jax.nn.initializers import lecun_normal

DType = Any
default_kernel_init = lecun_normal()
log_cosh = nk.nn.activation.log_cosh


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
    best_idx = jnp.argmin(norms2, axis=0, keepdims=True)
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


def _logdet_cmplx(matrix: jax.Array) -> jax.Array:
    # Written with Codex 02-18-26.
    sign, logabsdet = jnp.linalg.slogdet(matrix)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


class LogSlaterDeterminant(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    kernel_init: Any = default_kernel_init
    param_dtype: DType = complex

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        m_orbitals = self.param(
            "M",
            self.kernel_init,
            (self.hilbert.n_orbitals, self.hilbert.n_fermions),
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


class SpatialFactoredMultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6
    xi_init: float = 1.5

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


class SpatialEncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
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


class LogSpatialViT(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
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


class LogSlaterSpatialViT(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    num_layers: int
    d_model: int
    n_heads: int
    pair_classes: Any
    pair_distances: Any
    slater_kernel_init: Any = default_kernel_init
    slater_param_dtype: DType = complex
    vit_param_dtype: DType = jnp.float64
    mlp_hidden_factor: int = 4
    output_hidden_dim: int | None = None
    vit_kernel_init: Any = nn.initializers.xavier_uniform()
    xi_epsilon: float = 1.0e-6

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-18-26.
        log_slater = LogSlaterDeterminant(
            hilbert=self.hilbert,
            kernel_init=self.slater_kernel_init,
            param_dtype=self.slater_param_dtype,
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
