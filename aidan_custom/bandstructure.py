from __future__ import annotations

import numpy as np

from .geometry import ANNN_VECTORS, DELTA_VECTORS


def bloch_hamiltonian_at_k(
    kx: float,
    ky: float,
    t1: float = 1.0,
    t2: float = 0.15,
    phi: float = np.pi / 2,
    m: float = 0.0,
) -> np.ndarray:
    # Written with Codex 02-18-26.
    k = np.array([kx, ky])
    f = -t1 * sum(np.exp(1j * np.dot(k, d)) for d in DELTA_VECTORS)
    d0 = -2.0 * t2 * np.cos(phi) * sum(np.cos(np.dot(k, a)) for a in ANNN_VECTORS)
    dz = m + 2.0 * t2 * np.sin(phi) * sum(np.sin(np.dot(k, a)) for a in ANNN_VECTORS)
    return np.array([[d0 + dz, f], [np.conjugate(f), d0 - dz]], dtype=complex)


def eigenvalues_at_k(
    kx: float,
    ky: float,
    t1: float = 1.0,
    t2: float = 0.15,
    phi: float = np.pi / 2,
    m: float = 0.0,
) -> np.ndarray:
    # Written with Codex 02-18-26.
    return np.linalg.eigvalsh(
        bloch_hamiltonian_at_k(kx, ky, t1=t1, t2=t2, phi=phi, m=m)
    )


def dvector_and_derivatives_at_k(
    kx: float,
    ky: float,
    t1: float = 1.0,
    t2: float = 0.15,
    phi: float = np.pi / 2,
    m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    k = np.array([kx, ky])
    phase_nn = [np.exp(1j * np.dot(k, d)) for d in DELTA_VECTORS]

    f = -t1 * sum(phase_nn)
    df_dkx = -t1 * sum(1j * d[0] * p for d, p in zip(DELTA_VECTORS, phase_nn))
    df_dky = -t1 * sum(1j * d[1] * p for d, p in zip(DELTA_VECTORS, phase_nn))

    ddz_dkx = 2.0 * t2 * np.sin(phi) * sum(
        a[0] * np.cos(np.dot(k, a)) for a in ANNN_VECTORS
    )
    ddz_dky = 2.0 * t2 * np.sin(phi) * sum(
        a[1] * np.cos(np.dot(k, a)) for a in ANNN_VECTORS
    )

    dvec = np.array(
        [
            f.real,
            -f.imag,
            m + 2.0 * t2 * np.sin(phi) * sum(np.sin(np.dot(k, a)) for a in ANNN_VECTORS),
        ],
        dtype=float,
    )
    ddx = np.array([df_dkx.real, -df_dkx.imag, ddz_dkx], dtype=float)
    ddy = np.array([df_dky.real, -df_dky.imag, ddz_dky], dtype=float)
    return dvec, ddx, ddy


def qgt_from_bloch_hamiltonian(
    kx: float,
    ky: float,
    t1: float = 1.0,
    t2: float = 0.15,
    phi: float = np.pi / 2,
    m: float = 0.0,
    band: str = "lower",
) -> np.ndarray:
    # Written with Codex 02-18-26.
    dvec, ddx, ddy = dvector_and_derivatives_at_k(kx, ky, t1=t1, t2=t2, phi=phi, m=m)
    dnorm = np.linalg.norm(dvec)
    if dnorm < 1e-12:
        raise ValueError("Band touching (|d| ~ 0): QGT is singular at this k point.")

    dhat = dvec / dnorm
    ddhatx = ddx / dnorm - dvec * np.dot(dvec, ddx) / (dnorm**3)
    ddhaty = ddy / dnorm - dvec * np.dot(dvec, ddy) / (dnorm**3)

    g_xx = 0.25 * np.dot(ddhatx, ddhatx)
    g_yy = 0.25 * np.dot(ddhaty, ddhaty)
    g_xy = 0.25 * np.dot(ddhatx, ddhaty)

    berry_lower = -0.5 * np.dot(dhat, np.cross(ddhatx, ddhaty))
    if band == "lower":
        berry = berry_lower
    elif band == "upper":
        berry = -berry_lower
    else:
        raise ValueError("band must be 'lower' or 'upper'.")

    return np.array(
        [[g_xx, g_xy + 0.5j * berry], [g_xy - 0.5j * berry, g_yy]],
        dtype=complex,
    )


def berry_curvature_and_metric_trace_at_k(
    kx: float,
    ky: float,
    t1: float = 1.0,
    t2: float = 0.15,
    phi: float = np.pi / 2,
    m: float = 0.0,
    band: str = "lower",
) -> tuple[float, float]:
    # Written with Codex 02-18-26.
    qgt = qgt_from_bloch_hamiltonian(kx, ky, t1=t1, t2=t2, phi=phi, m=m, band=band)
    berry_curvature = 2.0 * float(np.imag(qgt[0, 1]))
    metric_trace = float(np.real(qgt[0, 0] + qgt[1, 1]))
    return berry_curvature, metric_trace
