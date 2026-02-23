"""Tests for PsiFormer neural quantum state architecture.

Three test groups:
  1. test_output_shape_dtype  – output is (batch,) complex.
  2. test_antisymmetry        – swapping two particle tokens flips det sign.
  3. test_noninteracting_energy – VMC on a small non-interacting MoTe2 system
                                  converges to the exact single-particle energy.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import pytest

from aidan_custom.bloch_ed import (
    build_mote2_three_orbital_lattice_embedding,
    build_mote2_three_orbital_real_space_terms,
)
from aidan_custom.models import (
    LogPsiFormer,
    NeuralOrbitalHead,
    make_supercell_reciprocal_vectors,
)
from aidan_custom.models.slater import _logdet_cmplx
from aidan_custom.mote2_three_orbital import MOTE2_A1, MOTE2_A2
from aidan_custom.mote2_three_orbital_model import build_mote2_three_orbital_hamiltonian


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# MoTe2 "ideal band" parameters used throughout the test suite.
_MOTE2_PARAMS = dict(
    delta=5.426542963374432,
    ez=0.0,
    t_th1=1.0,
    t_hh1=0.4689334070165771,
    t_th2=-0.07151471515876148,
    t_hh3=0.037340187144172164,
    t_tt1=-0.0377535889269079,
    a_m=1.0,
    ph_conj=True,
)


def _mote2_site_positions(Lx: int, Ly: int, a_m: float = 1.0) -> np.ndarray:
    """Real-space site positions for the MoTe2 three-orbital model."""
    lattice = build_mote2_three_orbital_lattice_embedding(Lx=Lx, Ly=Ly)
    n_sites = Lx * Ly * lattice.n_orbitals_per_cell
    a1 = a_m * np.asarray(MOTE2_A1, dtype=np.float64)
    a2 = a_m * np.asarray(MOTE2_A2, dtype=np.float64)
    u0 = np.array([a_m / np.sqrt(3.0), 0.0], dtype=np.float64)
    offsets = np.array([u0, np.zeros(2), -u0], dtype=np.float64)
    positions = np.empty((n_sites, 2), dtype=np.float64)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        positions[site] = float(x) * a1 + float(y) * a2 + offsets[int(orb)]
    return positions


def _hashable_matrix(arr: np.ndarray) -> tuple:
    arr = np.asarray(arr, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def _build_psiformer(hilbert, positions, g_vectors, **kwargs) -> LogPsiFormer:
    """Construct a small LogPsiFormer for testing."""
    defaults = dict(
        num_layers=1,
        d_model=8,
        n_heads=2,
        n_determinants=1,
        mlp_hidden_factor=2,
        param_dtype=jnp.float64,
    )
    defaults.update(kwargs)
    return LogPsiFormer(
        hilbert=hilbert,
        positions=_hashable_matrix(positions),
        g_vectors=_hashable_matrix(g_vectors),
        **defaults,
    )


# ---------------------------------------------------------------------------
# Test 1: output shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_dtype():
    """LogPsiFormer must return a complex array of shape (batch,)."""
    Lx, Ly, n_fermions = 1, 2, 2
    a_m = _MOTE2_PARAMS["a_m"]

    _, hi, _ = build_mote2_three_orbital_hamiltonian(
        Lx=Lx, Ly=Ly, n_fermions=n_fermions, V1=0.0, **_MOTE2_PARAMS
    )

    positions = _mote2_site_positions(Lx=Lx, Ly=Ly, a_m=a_m)
    basis_vectors = np.stack(
        [a_m * np.asarray(MOTE2_A1), a_m * np.asarray(MOTE2_A2)], axis=0
    )
    g_vectors = np.asarray(
        make_supercell_reciprocal_vectors(basis_vectors=basis_vectors, Lx=Lx, Ly=Ly),
        dtype=np.float64,
    )

    model = _build_psiformer(hi, positions, g_vectors)
    key = jax.random.PRNGKey(0)

    # Use a batch of 3 configurations from the Hilbert space.
    all_states = hi.all_states()
    batch = jnp.asarray(all_states[:3])

    params = model.init(key, batch)
    log_psi = model.apply(params, batch)

    assert log_psi.shape == (3,), f"Expected shape (3,), got {log_psi.shape}"
    assert jnp.issubdtype(log_psi.dtype, jnp.complexfloating), (
        f"Expected complex dtype, got {log_psi.dtype}"
    )
    assert jnp.all(jnp.isfinite(jnp.real(log_psi))), "Real parts contain non-finite values."
    assert jnp.all(jnp.isfinite(jnp.imag(log_psi))), "Imag parts contain non-finite values."


# ---------------------------------------------------------------------------
# Test 2: fermionic antisymmetry
# ---------------------------------------------------------------------------

def test_antisymmetry():
    """Swapping two particle input tokens must flip the determinant sign.

    NeuralOrbitalHead is permutation-equivariant: if we swap tokens p and p'
    in the input, rows p and p' of every output Slater matrix are swapped.
    An odd permutation of rows changes the sign of the determinant, so

        det(Φ_swapped) = -det(Φ)  i.e.  exp(log_det_swapped - log_det) = -1.

    This is the core mechanism that gives LogPsiFormer its fermionic
    antisymmetry without a separate Slater factor.
    """
    n_f = 3
    d_model = 8
    n_det = 1
    batch = 4

    head = NeuralOrbitalHead(
        n_particles=n_f,
        n_determinants=n_det,
        d_model=d_model,
        param_dtype=jnp.float64,
        kernel_init=nn_initializers_xavier_uniform(),
    )

    key_init, key_h = jax.random.split(jax.random.PRNGKey(42))
    # Random token embeddings: (batch, n_f, d_model).
    h = jax.random.normal(key_h, (batch, n_f, d_model), dtype=jnp.float64)

    params = head.init(key_init, h)
    orbital_matrices = head.apply(params, h)  # (batch, 1, n_f, n_f)

    # Swap particle tokens 0 and 1.
    h_swapped = h[:, [1, 0, 2], :]
    orbital_matrices_swapped = head.apply(params, h_swapped)

    log_det_fn = jnp.vectorize(_logdet_cmplx, signature="(n,n)->()")
    log_dets = log_det_fn(orbital_matrices[:, 0, :, :])          # (batch,) complex
    log_dets_swapped = log_det_fn(orbital_matrices_swapped[:, 0, :, :])  # (batch,)

    # exp(log_det_swapped - log_det) should be -1 for every batch element.
    ratio = jnp.exp(log_dets_swapped - log_dets)
    np.testing.assert_allclose(
        np.array(ratio.real), np.full(batch, -1.0), atol=1e-10,
        err_msg="Determinant ratio real part is not -1 after particle swap.",
    )
    np.testing.assert_allclose(
        np.array(ratio.imag), np.zeros(batch), atol=1e-10,
        err_msg="Determinant ratio imaginary part is not 0 after particle swap.",
    )


# Small shim so we can reference nn.initializers inside the test without
# importing flax at module level in a verbose way.
def nn_initializers_xavier_uniform():
    import flax.linen as nn
    return nn.initializers.xavier_uniform()


# ---------------------------------------------------------------------------
# Test 3: non-interacting ground-state energy
# ---------------------------------------------------------------------------

def test_noninteracting_energy():
    """PsiFormer converges to the exact non-interacting ground-state energy.

    System: MoTe2 three-orbital model, Lx=1, Ly=2, n_fermions=2, V1=0.
    Hilbert space: C(6, 2) = 15 states — tiny enough for FullSumState.

    The exact non-interacting energy is the sum of the two lowest eigenvalues
    of the one-body Hamiltonian matrix.  With FullSumState (exact expectation
    values, no MCMC noise) and Stochastic Reconfiguration, PsiFormer should
    reach this energy within 1 % relative error in ≤ 400 optimisation steps.
    """
    Lx, Ly, n_fermions = 1, 2, 2
    a_m = _MOTE2_PARAMS["a_m"]

    # --- Build non-interacting Hamiltonian ---
    graph, hi, ham = build_mote2_three_orbital_hamiltonian(
        Lx=Lx, Ly=Ly, n_fermions=n_fermions, V1=0.0, **_MOTE2_PARAMS
    )
    ham_sr = ham.to_fermionoperator2nd()

    # --- Exact non-interacting reference energy ---
    _, one_body_matrix, _ = build_mote2_three_orbital_real_space_terms(
        Lx=Lx, Ly=Ly, V1=0.0, **_MOTE2_PARAMS
    )
    evals = np.linalg.eigvalsh(np.asarray(one_body_matrix, dtype=np.complex128))
    exact_ni_energy = float(np.real(evals[:n_fermions].sum()))

    # --- PsiFormer model ---
    positions = _mote2_site_positions(Lx=Lx, Ly=Ly, a_m=a_m)
    basis_vectors = np.stack(
        [a_m * np.asarray(MOTE2_A1), a_m * np.asarray(MOTE2_A2)], axis=0
    )
    g_vectors = np.asarray(
        make_supercell_reciprocal_vectors(basis_vectors=basis_vectors, Lx=Lx, Ly=Ly),
        dtype=np.float64,
    )
    model = _build_psiformer(
        hi, positions, g_vectors,
        num_layers=2,
        d_model=16,
        n_heads=2,
        n_determinants=1,
    )

    # --- VMC with FullSumState (exact gradients, no MCMC noise) ---
    vstate = nk.vqs.FullSumState(hi, model, seed=0)

    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=0.01,
        mode="complex",
    )
    driver.run(n_iter=200)

    # Use ham_sr (FermionOperator2nd) for the final expect call; the
    # ParticleNumberConservingFermioperator2nd (ham) is not hashable by
    # FullSumState's sparse-matrix cache.
    final_energy = float(np.real(np.asarray(vstate.expect(ham_sr).mean)))

    # Variational bound: E_vmc >= E_exact (within numerical precision).
    assert final_energy >= exact_ni_energy - 1e-6, (
        f"VMC energy {final_energy:.8f} fell below exact energy "
        f"{exact_ni_energy:.8f} by more than numerical tolerance."
    )

    # Convergence check: within 1 % relative error.
    rel_error = abs(final_energy - exact_ni_energy) / max(abs(exact_ni_energy), 1e-10)
    assert rel_error < 0.01, (
        f"PsiFormer did not converge: "
        f"E_vmc={final_energy:.8f}, E_exact={exact_ni_energy:.8f}, "
        f"rel_error={rel_error:.4%}"
    )
