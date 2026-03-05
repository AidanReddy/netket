from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from ._common import DType
from .boseformer import BoseFormerEncoder
from .embedding import OccupiedPeriodicFeatureEmbedding, OccupiedSiteIndexEmbedding
from .slater import _hashable_matrix_literal, _logdet_cmplx


class NeuralOrbitalHead(nn.Module):
    """Maps particle token embeddings to K neural Slater orbital matrices.

    Each particle token h_p ∈ R^d_model is projected (independently and with
    shared weights) to K * n_particles complex orbital amplitudes.  Stacking
    over particles gives K square matrices of shape (n_particles, n_particles)
    whose determinants encode the many-body wavefunction.

    The head is permutation-equivariant in the particle dimension: swapping two
    input tokens swaps the corresponding rows of every output matrix, so the
    determinant acquires the correct fermionic sign flip.

    Args:
        n_particles:    Number of occupied particles N_f.
        n_determinants: Number of Slater matrices K.
        d_model:        Token embedding dimension.
        param_dtype:    Parameter dtype (real).
        kernel_init:    Weight initialiser for both linear projections.

    Input:
        x: (batch, n_particles, d_model)  real token embeddings.

    Output:
        Φ: (batch, n_determinants, n_particles, n_particles)  complex orbital
           matrices; Φ[b, k, p, j] is the amplitude of orbital j at particle p
           for determinant k in batch element b.
    """

    n_particles: int
    n_determinants: int
    d_model: int
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        # Written with Codex 02-22-26.
        out_dim = self.n_determinants * self.n_particles
        self.pre_head_ln = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="pre_head_ln",
        )
        self.real_dense = nn.Dense(
            out_dim,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="real_dense",
        )
        self.imag_dense = nn.Dense(
            out_dim,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            name="imag_dense",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Written with Codex 02-22-26.
        # x: (batch, n_particles, d_model)
        z = self.pre_head_ln(x)
        real = self.real_dense(z)  # (batch, n_particles, K * n_particles)
        imag = self.imag_dense(z)  # (batch, n_particles, K * n_particles)

        # Build complex orbital coefficients.
        orb = jax.lax.complex(real, imag)  # (batch, n_particles, K * n_particles)

        batch_size = x.shape[0]
        # Reshape to separate determinant and orbital indices.
        # orb[b, p, k, j] = amplitude of orbital j for particle p, determinant k.
        orb = orb.reshape(
            batch_size, self.n_particles, self.n_determinants, self.n_particles
        )
        # Transpose to (batch, K, n_particles, n_particles).
        # Φ[b, k, p, j]: row p = particle p, column j = orbital j.
        return jnp.transpose(orb, (0, 2, 1, 3))


class LogPsiFormer(nn.Module):
    """PsiFormer: Transformer-based neural multi-determinant Slater ansatz.

    Architecture
    ------------
    1. Periodic feature embedding: occupied site coordinates → token embeddings.
    2. BoseFormer encoder: permutation-equivariant self-attention over N_f tokens.
    3. Neural orbital head: each token → K × N_f complex amplitudes (the rows of
       K independent N_f × N_f Slater matrices).
    4. Determinant aggregation:
           Ψ(n) = Σ_k  c_k  det(Φ^(k)(n))
       computed stably in log-space via complex logsumexp.
    5. Optional Jastrow factor (NetKet's built-in two-body Jastrow).

    Fermionic antisymmetry is exact by construction: the BoseFormer encoder is
    permutation-equivariant, so swapping two particles swaps the corresponding
    rows of each Φ^(k), and det flips sign.  No separate Slater factor is
    needed.

    Args:
        hilbert:              SpinOrbitalFermions Hilbert space.
        positions:            (n_sites, 2) real-space site positions (hashable).
                              Required when embedding_type="periodic"; ignored otherwise.
        g_vectors:            (2, 2) supercell reciprocal vectors G1, G2 (hashable).
                              Required when embedding_type="periodic"; ignored otherwise.
        num_layers:           Number of BoseFormer encoder blocks.
        d_model:              Transformer model dimension.
        n_heads:              Number of attention heads (must divide d_model).
        n_determinants:       Number of Slater determinants K.  Default 1.
        mlp_hidden_factor:    MLP hidden-layer expansion factor.  Default 4.
        embedding_type:       "periodic" (default) — sin/cos periodic features from
                              real-space site coordinates; or "site_index" — a learned
                              embedding table indexed by integer site index (0-based).
        use_jastrow:          If True, add NetKet two-body Jastrow.  Default False.
        jastrow_param_dtype:  Jastrow parameter dtype.
        param_dtype:          Parameter dtype for encoder and orbital head.
        kernel_init:          Weight initialiser for encoder layers and embedding.
        orbital_kernel_init:  Weight initialiser for orbital head projections.
                              Xavier uniform is recommended (not zeros) because the
                              head feeds a determinant; an all-zero init gives a
                              singular matrix.
    """

    hilbert: nk.hilbert.SpinOrbitalFermions
    positions: Any  # (n_sites, 2); None is allowed when embedding_type="site_index"
    g_vectors: Any  # (2, 2);       None is allowed when embedding_type="site_index"
    num_layers: int
    d_model: int
    n_heads: int
    n_determinants: int = 1
    mlp_hidden_factor: int = 4
    embedding_type: str = "periodic"
    use_jastrow: bool = False
    jastrow_param_dtype: DType = jnp.float64
    param_dtype: DType = jnp.float64
    kernel_init: Any = nn.initializers.xavier_uniform()
    orbital_kernel_init: Any = nn.initializers.xavier_uniform()

    def __post_init__(self):
        # Written with Codex 02-22-26.
        object.__setattr__(self, "positions", _hashable_matrix_literal(self.positions))
        object.__setattr__(self, "g_vectors", _hashable_matrix_literal(self.g_vectors))
        super().__post_init__()

    @nn.compact
    def __call__(self, n: jax.Array) -> jax.Array:
        # Written with Codex 02-22-26.
        n_f = self.hilbert.n_fermions
        K = self.n_determinants

        # --- Input embedding (toggled by embedding_type) ---
        if self.embedding_type == "periodic":
            x = OccupiedPeriodicFeatureEmbedding(
                n_particles=n_f,
                positions=self.positions,
                g_vectors=self.g_vectors,
                d_model=self.d_model,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                name="embed",
            )(n)
        elif self.embedding_type == "site_index":
            x = OccupiedSiteIndexEmbedding(
                n_particles=n_f,
                n_sites=self.hilbert.size,
                d_model=self.d_model,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                name="embed",
            )(n)
        else:
            raise ValueError(
                f"Unknown embedding_type={self.embedding_type!r}. "
                "Choose 'periodic' or 'site_index'."
            )

        # --- BoseFormer encoder (permutation-equivariant) ---
        x = BoseFormerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_hidden_factor=self.mlp_hidden_factor,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            name="encoder",
        )(x)

        # --- Neural orbital head ---
        # orbital_matrices: (batch, K, N_f, N_f) complex
        orbital_matrices = NeuralOrbitalHead(
            n_particles=n_f,
            n_determinants=K,
            d_model=self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=self.orbital_kernel_init,
            name="orbital_head",
        )(x)

        # --- Log-determinants for each of the K matrices ---
        # jnp.vectorize broadcasts _logdet_cmplx over the leading (batch, K) axes.
        log_det_fn = jnp.vectorize(_logdet_cmplx, signature="(n,n)->()")
        log_dets = log_det_fn(orbital_matrices)  # (batch, K) complex

        # --- Learnable complex mixing coefficients c_k ---
        # Stored as real/imaginary parts; c = c_re + i c_im.
        # log|c| + i·arg(c) is computed via polar form to handle any c ∈ ℂ.
        c_re = self.param("det_coeffs_re", nn.initializers.ones, (K,), self.param_dtype)
        c_im = self.param("det_coeffs_im", nn.initializers.zeros, (K,), self.param_dtype)
        c = jax.lax.complex(c_re, c_im)  # (K,) complex
        log_c = jnp.log(jnp.abs(c)) + 1j * jnp.angle(c)  # (K,) complex

        # terms[b, k] = log(c_k) + log det(Φ^(k)(n_b))
        terms = log_dets + log_c[None, :]  # (batch, K) complex

        # Numerically stable complex logsumexp (stabilise over real part).
        m = jnp.real(terms).max(axis=-1, keepdims=True)  # (batch, 1) real
        log_psi = m[..., 0] + jnp.log(jnp.sum(jnp.exp(terms - m), axis=-1))

        # --- Optional two-body Jastrow ---
        if self.use_jastrow:
            log_jastrow = nk.models.Jastrow(
                param_dtype=self.jastrow_param_dtype,
                name="jastrow",
            )(n)
            log_psi = log_psi + log_jastrow

        return log_psi
