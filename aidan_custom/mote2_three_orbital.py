from __future__ import annotations

from functools import lru_cache

import numpy as np

from .geometry import reciprocal_vectors, sample_shortest_representative_bz

MOTE2_A1 = np.array([np.sqrt(3.0) / 2.0, 0.5], dtype=float)
MOTE2_A2 = np.array([-np.sqrt(3.0) / 2.0, 0.5], dtype=float)
MOTE2_A3 = np.array([0, -1], dtype=float)


@lru_cache(maxsize=None)
def _mote2_vectors(
    a_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    if a_m <= 0.0:
        raise ValueError(f"a_m must be positive, got {a_m}.")

    a1 = float(a_m) * MOTE2_A1
    a2 = float(a_m) * MOTE2_A2
    a3 = float(a_m) * MOTE2_A3
    a_vectors = np.stack((a1, a2, a3), axis=0)

    # Literal bond-vector gauge from Fig. 2(b), with u_0 to the right.
    # u_j follow the convention set by the current A-vector definitions above.
    u0 = np.array([1,0]) / np.sqrt(3) * float(a_m)
    u1 = np.array([-1/2, +np.sqrt(3)/2]) /np.sqrt(3) * float(a_m)
    u2 = np.array([-1/2, -np.sqrt(3)/2]) /np.sqrt(3) * float(a_m)

    u_vectors = np.stack((u0, u1, u2), axis=0)

    return a1, a2, a3, a_vectors, u_vectors


def _phase_forms_and_gradients(
    kvec: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
) -> tuple[complex, complex, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    phase0 = np.exp(1j * np.dot(u0, kvec))
    phase1 = np.exp(1j * np.dot(u1, kvec))
    phase2 = np.exp(1j * np.dot(u2, kvec))

    f_k = phase0 + phase1 + phase2

    # c3_j = exp[2 pi i (1-j)/3], j = 0,1,2
    c3_0 = np.exp(2j * np.pi / 3.0)
    c3_1 = 1.0 + 0.0j
    c3_2 = np.exp(-2j * np.pi / 3.0)
    g_k = c3_0 * phase0 + c3_1 * phase1 + c3_2 * phase2

    df_dq = 1j * (phase0 * u0 + phase1 * u1 + phase2 * u2)
    dg_dq = 1j * (c3_0 * phase0 * u0 + c3_1 * phase1 * u1 + c3_2 * phase2 * u2)
    return f_k, g_k, df_dq, dg_dq


def _hk_and_gradient(kvec: np.ndarray, a_vectors: np.ndarray) -> tuple[float, np.ndarray]:
    # Written with Codex 02-20-26.
    kdot = a_vectors @ kvec
    h_k = 2.0 * float(np.sum(np.cos(kdot), dtype=float))
    dh_dk = -2.0 * np.sum(a_vectors * np.sin(kdot)[:, None], axis=0, dtype=float)
    return h_k, dh_dk


def _mote2_hamiltonian_and_derivatives(
    kx: float,
    ky: float,
    delta: float,
    ez: float,
    t_th1: float,
    t_hh1: float,
    t_th2: float,
    t_hh3: float,
    t_tt1: float,
    a_m: float,
    ph_conj: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    ph_conj = bool(ph_conj)
    if ph_conj:
        kvec = np.array([-kx, -ky], dtype=float)
    else:
        kvec = np.array([kx, ky], dtype=float)
    _, _, _, a_vectors, u_vectors = _mote2_vectors(float(a_m))
    u0 = u_vectors[0]
    u1 = u_vectors[1]
    u2 = u_vectors[2]

    f_k, g_k, df_k, dg_k = _phase_forms_and_gradients(kvec, u0, u1, u2)

    _, g_mk, _, dg_mk_q = _phase_forms_and_gradients(-kvec, u0, u1, u2)
    dg_mk = -dg_mk_q

    f_2k, g_2k, df_2k_q, dg_2k_q = _phase_forms_and_gradients(2.0 * kvec, u0, u1, u2)
    df_2k = 2.0 * df_2k_q
    dg_2k = 2.0 * dg_2k_q

    _, g_m2k, _, dg_m2k_q = _phase_forms_and_gradients(-2.0 * kvec, u0, u1, u2)
    dg_m2k = -2.0 * dg_m2k_q

    h_k, dh_k = _hk_and_gradient(kvec, a_vectors)

    h = np.zeros((3, 3), dtype=np.complex128)
    dhx = np.zeros((3, 3), dtype=np.complex128)
    dhy = np.zeros((3, 3), dtype=np.complex128)

    # Build Eq. (2) explicitly as h_pot + h_nn + h_lr.
    h_pot = np.zeros((3, 3), dtype=np.complex128)
    h_pot[0, 0] = float(ez)
    h_pot[1, 1] = -float(delta)
    h_pot[2, 2] = -float(ez)

    h_nn = np.zeros((3, 3), dtype=np.complex128)
    h_nn[0, 1] = -t_th1 * g_k
    h_nn[1, 0] = -t_th1 * np.conjugate(g_k)
    h_nn[0, 2] = t_hh1 * np.conjugate(f_k)
    h_nn[2, 0] = t_hh1 * f_k
    h_nn[1, 2] = t_th1 * np.conjugate(g_mk)
    h_nn[2, 1] = t_th1 * g_mk

    h_lr = np.zeros((3, 3), dtype=np.complex128)
    h_lr[0, 1] = t_th2 * g_m2k
    h_lr[1, 0] = t_th2 * np.conjugate(g_m2k)
    h_lr[0, 2] = t_hh3 * f_2k
    h_lr[2, 0] = t_hh3 * np.conjugate(f_2k)
    h_lr[1, 1] = t_tt1 * h_k
    h_lr[1, 2] = -t_th2 * np.conjugate(g_2k)
    h_lr[2, 1] = -t_th2 * g_2k

    h = h_pot + h_nn + h_lr

    dh_nn = (np.zeros((3, 3), dtype=np.complex128), np.zeros((3, 3), dtype=np.complex128))
    dh_lr = (np.zeros((3, 3), dtype=np.complex128), np.zeros((3, 3), dtype=np.complex128))

    # x-component derivatives.
    dh_nn[0][0, 1] = -t_th1 * dg_k[0]
    dh_nn[0][1, 0] = -t_th1 * np.conjugate(dg_k[0])
    dh_nn[0][0, 2] = t_hh1 * np.conjugate(df_k[0])
    dh_nn[0][2, 0] = t_hh1 * df_k[0]
    dh_nn[0][1, 2] = t_th1 * np.conjugate(dg_mk[0])
    dh_nn[0][2, 1] = t_th1 * dg_mk[0]

    dh_lr[0][0, 1] = t_th2 * dg_m2k[0]
    dh_lr[0][1, 0] = t_th2 * np.conjugate(dg_m2k[0])
    dh_lr[0][0, 2] = t_hh3 * df_2k[0]
    dh_lr[0][2, 0] = t_hh3 * np.conjugate(df_2k[0])
    dh_lr[0][1, 1] = t_tt1 * dh_k[0]
    dh_lr[0][1, 2] = -t_th2 * np.conjugate(dg_2k[0])
    dh_lr[0][2, 1] = -t_th2 * dg_2k[0]

    # y-component derivatives.
    dh_nn[1][0, 1] = -t_th1 * dg_k[1]
    dh_nn[1][1, 0] = -t_th1 * np.conjugate(dg_k[1])
    dh_nn[1][0, 2] = t_hh1 * np.conjugate(df_k[1])
    dh_nn[1][2, 0] = t_hh1 * df_k[1]
    dh_nn[1][1, 2] = t_th1 * np.conjugate(dg_mk[1])
    dh_nn[1][2, 1] = t_th1 * dg_mk[1]

    dh_lr[1][0, 1] = t_th2 * dg_m2k[1]
    dh_lr[1][1, 0] = t_th2 * np.conjugate(dg_m2k[1])
    dh_lr[1][0, 2] = t_hh3 * df_2k[1]
    dh_lr[1][2, 0] = t_hh3 * np.conjugate(df_2k[1])
    dh_lr[1][1, 1] = t_tt1 * dh_k[1]
    dh_lr[1][1, 2] = -t_th2 * np.conjugate(dg_2k[1])
    dh_lr[1][2, 1] = -t_th2 * dg_2k[1]

    dhx = dh_nn[0] + dh_lr[0]
    dhy = dh_nn[1] + dh_lr[1]

    if ph_conj:
        # h_ph(k) = -h*(-k); gradients are +conj(∂h/∂k_eval) at k_eval=-k.
        h = -np.conjugate(h)
        dhx = np.conjugate(dhx)
        dhy = np.conjugate(dhy)

    return h, dhx, dhy


def _resolve_band_index(band: int | str, n_bands: int) -> int:
    # Written with Codex 02-20-26.
    if isinstance(band, str):
        key = band.lower().strip()
        if key in {"lowest", "lower", "valence", "0"}:
            return 0
        if key in {"middle", "1"}:
            return 1
        if key in {"highest", "upper", "conduction", "2"}:
            return n_bands - 1
        raise ValueError(
            "band must be an integer index or one of "
            "{'lowest','middle','highest','lower','upper','valence','conduction'}."
        )

    idx = int(band)
    if idx < 0:
        idx = n_bands + idx
    if not (0 <= idx < n_bands):
        raise ValueError(f"band index out of range: band={band}, n_bands={n_bands}.")
    return idx


def mote2_three_orbital_default_parameters() -> dict[str, float]:
    # Written with Codex 02-20-26.
    return {
        "delta": 0.0,
        "ez": 0.0,
        "t_th1": 1.0,
        "t_hh1": 1.0,
        "t_th2": 0.0,
        "t_hh3": 0.0,
        "t_tt1": 0.0,
        "a_m": 1.0,
    }


def mote2_three_orbital_parameters_from_t123(
    t1: float,
    t2: float,
    t3: float,
    delta: float = 0.0,
    ez: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
) -> dict[str, float]:
    # Written with Codex 02-20-26.
    return {
        "delta": float(delta),
        "ez": float(ez),
        "t_th1": float(t1),
        "t_hh1": float(t1),
        "t_th2": float(t2),
        "t_hh3": float(t3),
        "t_tt1": float(t_tt1),
        "a_m": float(a_m),
    }


def mote2_three_orbital_reciprocal_vectors(a_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-20-26.
    a1, a2, _, _, _ = _mote2_vectors(float(a_m))
    return reciprocal_vectors(a1, a2)


def mote2_three_orbital_bloch_hamiltonian(
    kx: float,
    ky: float,
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
    h, _, _ = _mote2_hamiltonian_and_derivatives(
        kx=float(kx),
        ky=float(ky),
        delta=float(delta),
        ez=float(ez),
        t_th1=float(t_th1),
        t_hh1=float(t_hh1),
        t_th2=float(t_th2),
        t_hh3=float(t_hh3),
        t_tt1=float(t_tt1),
        a_m=float(a_m),
        ph_conj=bool(ph_conj),
    )
    return h


def mote2_three_orbital_eigenvalues(
    kx: float,
    ky: float,
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
    return np.linalg.eigvalsh(
        mote2_three_orbital_bloch_hamiltonian(
            kx=kx,
            ky=ky,
            delta=delta,
            ez=ez,
            t_th1=t_th1,
            t_hh1=t_hh1,
            t_th2=t_th2,
            t_hh3=t_hh3,
            t_tt1=t_tt1,
            a_m=a_m,
            ph_conj=ph_conj,
        )
    )


def mote2_three_orbital_bloch_hamiltonian_t123(
    kx: float,
    ky: float,
    t1: float,
    t2: float,
    t3: float,
    delta: float = 0.0,
    ez: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    params = mote2_three_orbital_parameters_from_t123(
        t1=t1,
        t2=t2,
        t3=t3,
        delta=delta,
        ez=ez,
        t_tt1=t_tt1,
        a_m=a_m,
    )
    return mote2_three_orbital_bloch_hamiltonian(
        kx=kx,
        ky=ky,
        ph_conj=ph_conj,
        **params,
    )


def mote2_three_orbital_eigenvalues_t123(
    kx: float,
    ky: float,
    t1: float,
    t2: float,
    t3: float,
    delta: float = 0.0,
    ez: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
) -> np.ndarray:
    # Written with Codex 02-20-26.
    return np.linalg.eigvalsh(
        mote2_three_orbital_bloch_hamiltonian_t123(
            kx=kx,
            ky=ky,
            t1=t1,
            t2=t2,
            t3=t3,
            delta=delta,
            ez=ez,
            t_tt1=t_tt1,
            a_m=a_m,
            ph_conj=ph_conj,
        )
    )


def mote2_three_orbital_qgt(
    kx: float,
    ky: float,
    band: int | str = "lowest",
    dk: float = 1.0e-5,
    gap_tolerance: float = 1.0e-10,
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
    _ = dk
    h, dhx, dhy = _mote2_hamiltonian_and_derivatives(
        kx=float(kx),
        ky=float(ky),
        delta=float(delta),
        ez=float(ez),
        t_th1=float(t_th1),
        t_hh1=float(t_hh1),
        t_th2=float(t_th2),
        t_hh3=float(t_hh3),
        t_tt1=float(t_tt1),
        a_m=float(a_m),
        ph_conj=bool(ph_conj),
    )

    evals, evecs = np.linalg.eigh(h)
    n_bands = evals.size
    n = _resolve_band_index(band, n_bands)

    dhs = (dhx, dhy)
    qgt = np.zeros((2, 2), dtype=np.complex128)

    vec_n = evecs[:, n]
    for m in range(n_bands):
        if m == n:
            continue

        gap = float(abs(evals[n] - evals[m]))
        if gap < gap_tolerance:
            raise ValueError(
                f"Band gap is below tolerance at k=({kx},{ky}), bands {n} and {m}: gap={gap}."
            )

        vec_m = evecs[:, m]
        denom = (evals[n] - evals[m]) ** 2
        for mu in range(2):
            left = np.vdot(vec_n, dhs[mu] @ vec_m)
            for nu in range(2):
                right = np.vdot(vec_m, dhs[nu] @ vec_n)
                qgt[mu, nu] += left * right / denom

    return 0.5 * (qgt + np.conjugate(qgt.T))


def mote2_three_orbital_berry_and_metric_trace(
    kx: float,
    ky: float,
    band: int | str = "lowest",
    dk: float = 1.0e-5,
    gap_tolerance: float = 1.0e-10,
    delta: float = 0.0,
    ez: float = 0.0,
    t_th1: float = 1.0,
    t_hh1: float = 1.0,
    t_th2: float = 0.0,
    t_hh3: float = 0.0,
    t_tt1: float = 0.0,
    a_m: float = 1.0,
    ph_conj: bool = False,
) -> tuple[float, float]:
    # Written with Codex 02-20-26.
    qgt = mote2_three_orbital_qgt(
        kx=kx,
        ky=ky,
        band=band,
        dk=dk,
        gap_tolerance=gap_tolerance,
        delta=delta,
        ez=ez,
        t_th1=t_th1,
        t_hh1=t_hh1,
        t_th2=t_th2,
        t_hh3=t_hh3,
        t_tt1=t_tt1,
        a_m=a_m,
        ph_conj=ph_conj,
    )
    berry_curvature = 2.0 * float(np.imag(qgt[0, 1]))
    metric_trace = float(np.real(qgt[0, 0] + qgt[1, 1]))
    return berry_curvature, metric_trace


def mote2_three_orbital_chern_numbers(
    nx: int = 51,
    ny: int = 51,
    overlap_tolerance: float = 1.0e-14,
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
    # Compute Chern numbers by integrating Berry curvature from the QGT
    # (not by link-variable/Wilson-loop plaquettes), which is compatible
    # with the literal bond-vector gauge convention used in this module.
    nx = int(nx)
    ny = int(ny)
    if nx <= 1 or ny <= 1:
        raise ValueError(f"nx and ny must be > 1, got nx={nx}, ny={ny}.")
    gap_tolerance = float(overlap_tolerance)

    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=a_m)
    area_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    prefactor = area_bz / (2.0 * np.pi * nx * ny)
    chern = np.zeros(3, dtype=float)
    k_points = sample_shortest_representative_bz(nx, ny, b1, b2, unique=False)
    for kvec in k_points:
        for n in range(3):
            qgt = mote2_three_orbital_qgt(
                kx=float(kvec[0]),
                ky=float(kvec[1]),
                band=n,
                gap_tolerance=gap_tolerance,
                delta=delta,
                ez=ez,
                t_th1=t_th1,
                t_hh1=t_hh1,
                t_th2=t_th2,
                t_hh3=t_hh3,
                t_tt1=t_tt1,
                a_m=a_m,
                ph_conj=ph_conj,
            )
            chern[n] += 2.0 * float(np.imag(qgt[0, 1]))

    return prefactor * chern
