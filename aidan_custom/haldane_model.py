from __future__ import annotations

import numpy as np
import netket as nk
import netket.experimental as nkx
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc


def _site_id(graph: nk.graph.Honeycomb, x: int, y: int, sublattice: int, lx: int, ly: int) -> int:
    # Written with Codex 02-18-26.
    return int(graph.id_from_basis_coords([x % lx, y % ly, sublattice]))


def build_haldane_hamiltonian(
    Lx: int,
    Ly: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    V1: float = 0.0,
    n_fermions: int | None = None,
):
    # Written with Codex 02-18-26.
    graph = nk.graph.Honeycomb(extent=[Lx, Ly], pbc=True)
    n_sites = graph.n_nodes

    if n_fermions is None:
        n_fermions = n_sites // 2
    if not (0 <= n_fermions <= n_sites):
        raise ValueError(
            f"Invalid n_fermions={n_fermions} for spinless system with n_sites={n_sites}."
        )

    hi = nk.hilbert.SpinOrbitalFermions(n_sites, s=None, n_fermions=n_fermions)

    ham = 0.0 + 0.0j
    for i, j in graph.edges():
        ham += -t1 * (cdag(hi, i) @ c(hi, j) + cdag(hi, j) @ c(hi, i))
        ham += V1 * (nc(hi, i) @ nc(hi, j))

    phase = np.exp(1j * phi)
    nnn_shifts = ((-1, 1), (0, -1), (1, 0))
    for x in range(Lx):
        for y in range(Ly):
            a = _site_id(graph, x, y, 0, Lx, Ly)
            b = _site_id(graph, x, y, 1, Lx, Ly)
            for dx, dy in nnn_shifts:
                a2 = _site_id(graph, x + dx, y + dy, 0, Lx, Ly)
                b2 = _site_id(graph, x + dx, y + dy, 1, Lx, Ly)
                ham += -t2 * (
                    phase * (cdag(hi, a) @ c(hi, a2))
                    + phase.conjugate() * (cdag(hi, a2) @ c(hi, a))
                )
                ham += -t2 * (
                    phase.conjugate() * (cdag(hi, b) @ c(hi, b2))
                    + phase * (cdag(hi, b2) @ c(hi, b))
                )
            ham += m * (nc(hi, a) - nc(hi, b))

    ham = ham.reduce()
    ham = nkx.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(ham)
    return graph, hi, ham


def build_haldane_one_body_matrix(
    graph: nk.graph.Honeycomb,
    Lx: int,
    Ly: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    n_sites = graph.n_nodes
    h1 = np.zeros((n_sites, n_sites), dtype=np.complex128)

    for i, j in graph.edges():
        h1[i, j] += -t1
        h1[j, i] += -t1

    phase = np.exp(1j * phi)
    nnn_shifts = ((-1, 1), (0, -1), (1, 0))
    for x in range(Lx):
        for y in range(Ly):
            a = _site_id(graph, x, y, 0, Lx, Ly)
            b = _site_id(graph, x, y, 1, Lx, Ly)
            for dx, dy in nnn_shifts:
                a2 = _site_id(graph, x + dx, y + dy, 0, Lx, Ly)
                b2 = _site_id(graph, x + dx, y + dy, 1, Lx, Ly)
                h1[a, a2] += -t2 * phase
                h1[a2, a] += -t2 * phase.conjugate()
                h1[b, b2] += -t2 * phase.conjugate()
                h1[b2, b] += -t2 * phase
            h1[a, a] += m
            h1[b, b] += -m

    return h1


def noninteracting_slater_orbitals_haldane(
    graph: nk.graph.Honeycomb,
    Lx: int,
    Ly: int,
    n_fermions: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
) -> np.ndarray:
    # Written with Codex 02-19-26.
    h1 = build_haldane_one_body_matrix(
        graph=graph,
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
    )
    n_sites = h1.shape[0]
    if not (0 <= int(n_fermions) <= int(n_sites)):
        raise ValueError(
            f"Invalid n_fermions={n_fermions} for one-body matrix with n_sites={n_sites}."
        )

    _, eigvecs = np.linalg.eigh(h1)
    return np.asarray(eigvecs[:, :n_fermions], dtype=np.complex128)
