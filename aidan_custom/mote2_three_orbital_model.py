from __future__ import annotations

import numpy as np
import netket as nk
import netket.experimental as nkx
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc

from .bloch_ed import build_mote2_three_orbital_real_space_terms
from .bloch_ed.types import QuarticTermList


def _validate_n_fermions(n_sites: int, n_fermions: int | None) -> int:
    # Written with Codex 02-20-26.
    if n_fermions is None:
        n_fermions = int(n_sites) // 2
    n_fermions = int(n_fermions)
    if not (0 <= n_fermions <= int(n_sites)):
        raise ValueError(
            f"Invalid n_fermions={n_fermions} for spinless system with n_sites={n_sites}."
        )
    return n_fermions


def _graph_edges_from_real_space_terms(
    one_body_real_space: np.ndarray,
    site_terms: QuarticTermList,
    cutoff: float,
) -> list[tuple[int, int]]:
    # Written with Codex 02-20-26.
    matrix = np.asarray(one_body_real_space, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("one_body_real_space must be a square matrix.")

    n_sites = int(matrix.shape[0])
    cutoff = float(cutoff)
    edges: set[tuple[int, int]] = set()

    nz = np.argwhere(np.abs(matrix) > cutoff)
    for i_raw, j_raw in nz:
        i = int(i_raw)
        j = int(j_raw)
        if i == j:
            continue
        if i < j:
            edges.add((i, j))
        else:
            edges.add((j, i))

    n_terms = int(site_terms.coefficients.size)
    for idx in range(n_terms):
        coeff = np.complex128(site_terms.coefficients[idx])
        if abs(coeff) <= cutoff:
            continue
        i = int(site_terms.create_1[idx])
        j = int(site_terms.create_2[idx])
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise ValueError(
                "Site index out of range in quartic terms: "
                f"(i, j)=({i}, {j}) for n_sites={n_sites}."
            )
        if i == j:
            continue
        if i < j:
            edges.add((i, j))
        else:
            edges.add((j, i))

    return sorted(edges)


def build_mote2_three_orbital_hamiltonian(
    Lx: int,
    Ly: int,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
    V1: float = 0.0,
    n_fermions: int | None = None,
    operator_cutoff: float = 1e-12,
    graph_cutoff: float = 1e-12,
):
    # Written with Codex 02-20-26.
    lattice, one_body_real_space, site_terms = build_mote2_three_orbital_real_space_terms(
        Lx=Lx,
        Ly=Ly,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
        V1=V1,
    )
    del lattice

    one_body_real_space = np.asarray(one_body_real_space, dtype=np.complex128)
    n_sites = int(one_body_real_space.shape[0])
    n_fermions = _validate_n_fermions(n_sites=n_sites, n_fermions=n_fermions)

    edges = _graph_edges_from_real_space_terms(
        one_body_real_space=one_body_real_space,
        site_terms=site_terms,
        cutoff=float(graph_cutoff),
    )
    graph = nk.graph.Graph(edges=edges, n_nodes=n_sites)
    hi = nk.hilbert.SpinOrbitalFermions(n_sites, s=None, n_fermions=n_fermions)

    operator_cutoff = float(operator_cutoff)
    ham = 0.0 + 0.0j

    nz = np.argwhere(np.abs(one_body_real_space) > operator_cutoff)
    for i_raw, j_raw in nz:
        i = int(i_raw)
        j = int(j_raw)
        coeff = np.complex128(one_body_real_space[i, j])
        ham += coeff * (cdag(hi, i) @ c(hi, j))

    n_terms = int(site_terms.coefficients.size)
    for idx in range(n_terms):
        coeff = np.complex128(site_terms.coefficients[idx])
        if abs(coeff) <= operator_cutoff:
            continue
        a = int(site_terms.create_1[idx])
        b = int(site_terms.create_2[idx])
        c_idx = int(site_terms.annihilate_1[idx])
        d = int(site_terms.annihilate_2[idx])

        if a == c_idx and b == d:
            ham += coeff * (nc(hi, a) @ nc(hi, b))
        else:
            ham += coeff * (
                cdag(hi, a)
                @ cdag(hi, b)
                @ c(hi, d)
                @ c(hi, c_idx)
            )

    ham = ham.reduce()
    ham = nkx.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(ham)
    return graph, hi, ham


def noninteracting_slater_orbitals_mote2_three_orbital(
    Lx: int,
    Ly: int,
    n_fermions: int,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    _, one_body_real_space, _ = build_mote2_three_orbital_real_space_terms(
        Lx=Lx,
        Ly=Ly,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
        V1=0.0,
    )
    one_body_real_space = np.asarray(one_body_real_space, dtype=np.complex128)

    n_sites = int(one_body_real_space.shape[0])
    n_fermions = _validate_n_fermions(n_sites=n_sites, n_fermions=n_fermions)

    _, eigvecs = np.linalg.eigh(one_body_real_space)
    return np.asarray(eigvecs[:, :n_fermions], dtype=np.complex128)
