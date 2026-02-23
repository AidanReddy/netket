from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..mote2_three_orbital import _mote2_vectors, mote2_three_orbital_reciprocal_vectors


def mote2_site_positions_and_supercell_vectors(
    lattice,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    a1, a2, _, _, u_vectors = _mote2_vectors(float(a_m))
    u0 = np.asarray(u_vectors[0], dtype=np.float64)
    orbital_offsets = np.asarray(
        (
            u0,
            np.zeros(2, dtype=np.float64),
            -u0,
        ),
        dtype=np.float64,
    )

    n_sites = int(lattice.Lx) * int(lattice.Ly) * int(lattice.n_orbitals_per_cell)
    positions = np.empty((n_sites, 2), dtype=np.float64)
    for site in range(n_sites):
        x, y, orb = lattice.site_to_cell[site]
        bravais_coeff = lattice.cell_to_bravais[int(x), int(y)]
        positions[site] = (
            float(bravais_coeff[0]) * np.asarray(a1, dtype=np.float64)
            + float(bravais_coeff[1]) * np.asarray(a2, dtype=np.float64)
            + orbital_offsets[int(orb)]
        )

    supercell_t1 = (
        float(lattice.supercell_matrix[0, 0]) * np.asarray(a1, dtype=np.float64)
        + float(lattice.supercell_matrix[1, 0]) * np.asarray(a2, dtype=np.float64)
    )
    supercell_t2 = (
        float(lattice.supercell_matrix[0, 1]) * np.asarray(a1, dtype=np.float64)
        + float(lattice.supercell_matrix[1, 1]) * np.asarray(a2, dtype=np.float64)
    )
    return positions, supercell_t1, supercell_t2


def minimum_image_distance(
    delta: np.ndarray,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> float:
    # Written with Codex 02-21-26.
    best = np.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            trial = delta + float(n1) * supercell_t1 + float(n2) * supercell_t2
            norm = float(np.linalg.norm(trial))
            if norm < best:
                best = norm
    return best


def pair_correlation_from_corr_matrix(
    corr_matrix: np.ndarray,
    site_positions: np.ndarray,
    n_fermions: int,
    supercell_t1: np.ndarray,
    supercell_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    n_sites = int(corr_matrix.shape[0])
    dist_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    for i in range(n_sites):
        for j in range(n_sites):
            delta = site_positions[j] - site_positions[i]
            dist_matrix[i, j] = minimum_image_distance(
                delta=delta,
                supercell_t1=supercell_t1,
                supercell_t2=supercell_t2,
            )

    offdiag = ~np.eye(n_sites, dtype=bool)
    shells = np.round(dist_matrix, decimals=12)
    r_values = np.unique(shells[offdiag])
    norm_density = (float(n_fermions) / float(n_sites)) ** 2

    g_r = np.empty(r_values.size, dtype=np.float64)
    for idx, r_shell in enumerate(r_values):
        mask = (shells == r_shell) & offdiag
        g_r[idx] = float(np.real(corr_matrix[mask].mean())) / norm_density

    if r_values.size == 0 or not np.isclose(r_values[0], 0.0):
        r_plot = np.concatenate((np.asarray([0.0]), r_values))
        g_plot = np.concatenate((np.asarray([0.0]), g_r))
    else:
        r_plot = r_values.copy()
        g_plot = g_r.copy()
        g_plot[0] = 0.0

    return r_plot, g_plot


def q_vectors_from_lattice_mote2(
    lattice,
    a_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    b1, b2 = mote2_three_orbital_reciprocal_vectors(a_m=float(a_m))
    q_vectors = np.empty((int(lattice.Lx) * int(lattice.Ly), 2), dtype=np.float64)
    q_sector_indices = np.empty((int(lattice.Lx) * int(lattice.Ly), 2), dtype=np.int64)

    out = 0
    for kx in range(int(lattice.Lx)):
        for ky in range(int(lattice.Ly)):
            coeff = lattice.kpoint_coefficients[kx, ky]
            q_vectors[out] = coeff[0] * b1 + coeff[1] * b2
            q_sector_indices[out] = np.asarray([kx, ky], dtype=np.int64)
            out += 1

    return q_vectors, q_sector_indices


def structure_factor_from_corr_matrix(
    corr_matrix: np.ndarray,
    site_positions: np.ndarray,
    q_vectors: np.ndarray,
    n_fermions: int,
) -> np.ndarray:
    # Written with Codex 02-21-26.
    disp = site_positions[:, np.newaxis, :] - site_positions[np.newaxis, :, :]
    phase = np.exp(-1j * np.tensordot(disp, q_vectors, axes=([2], [1])))
    s_q = np.einsum("ij,ijq->q", corr_matrix, phase, optimize=True) / float(n_fermions)
    return np.real(s_q)


def radial_average(
    values: np.ndarray,
    vectors: np.ndarray,
    zero_mode_to_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Written with Codex 02-21-26.
    radii = np.linalg.norm(vectors, axis=1)
    vals = np.asarray(values, dtype=np.float64).copy()
    if zero_mode_to_zero:
        vals[radii < 1.0e-12] = 0.0

    shells = np.round(radii, decimals=12)
    unique_shells = np.unique(shells)
    shell_mean = np.empty(unique_shells.size, dtype=np.float64)
    shell_err = np.empty(unique_shells.size, dtype=np.float64)

    for idx, shell in enumerate(unique_shells):
        shell_vals = vals[shells == shell]
        shell_mean[idx] = float(np.mean(shell_vals))
        if shell_vals.size > 1:
            shell_err[idx] = float(np.std(shell_vals, ddof=1) / np.sqrt(shell_vals.size))
        else:
            shell_err[idx] = 0.0

    return unique_shells, shell_mean, shell_err


def write_spectrum_csv(path: Path, spectrum_rows: list[dict[str, float]]) -> None:
    # Written with Codex 02-21-26.
    fieldnames = [
        "sector_kx",
        "sector_ky",
        "kvec_x",
        "kvec_y",
        "eig_index",
        "energy",
        "sector_dimension",
        "solver",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(spectrum_rows)


def plot_k_resolved_spectrum(
    spectrum_rows: list[dict[str, float]],
    lattice_shape: tuple[int, int],
    output_path: Path,
) -> None:
    # Written with Codex 02-21-26.
    Lx, Ly = int(lattice_shape[0]), int(lattice_shape[1])
    sector_linear = np.asarray(
        [int(row["sector_kx"]) * Ly + int(row["sector_ky"]) for row in spectrum_rows],
        dtype=np.int64,
    )
    eig_index = np.asarray([int(row["eig_index"]) for row in spectrum_rows], dtype=np.int64)
    energies = np.asarray([float(row["energy"]) for row in spectrum_rows], dtype=np.float64)
    kx_vals = np.asarray([float(row["kvec_x"]) for row in spectrum_rows], dtype=np.float64)
    ky_vals = np.asarray([float(row["kvec_y"]) for row in spectrum_rows], dtype=np.float64)
    sector_kx = np.asarray([int(row["sector_kx"]) for row in spectrum_rows], dtype=np.int64)
    sector_ky = np.asarray([int(row["sector_ky"]) for row in spectrum_rows], dtype=np.int64)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    cmap = plt.cm.get_cmap("viridis", int(np.max(eig_index)) + 1)
    for eig_idx in range(int(np.max(eig_index)) + 1):
        mask = eig_index == eig_idx
        if np.any(mask):
            ax0.scatter(
                sector_linear[mask],
                energies[mask],
                s=12.0,
                color=cmap(eig_idx),
                label=f"n={eig_idx}",
                alpha=0.85,
            )

    ax0.set_title("k-resolved energy spectrum (10 lowest per sector)")
    ax0.set_xlabel("Momentum sector index")
    ax0.set_ylabel("Energy")
    xticks = np.arange(Lx * Ly, dtype=np.int64)
    step = max(1, (Lx * Ly) // 9)
    shown = xticks[::step]
    ax0.set_xticks(shown)
    ax0.set_xticklabels([f"{int(s // Ly)},{int(s % Ly)}" for s in shown], rotation=40, ha="right")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=7, ncol=2)

    lowest_mask = eig_index == 0
    sc = ax1.scatter(
        kx_vals[lowest_mask],
        ky_vals[lowest_mask],
        c=energies[lowest_mask],
        s=90.0,
        cmap="plasma",
        edgecolors="black",
        linewidths=0.3,
    )
    for xk, yk, kx_s, ky_s in zip(
        kx_vals[lowest_mask],
        ky_vals[lowest_mask],
        sector_kx[lowest_mask],
        sector_ky[lowest_mask],
    ):
        ax1.text(float(xk), float(yk), f"({int(kx_s)},{int(ky_s)})", fontsize=6, ha="left", va="bottom")

    ax1.set_title("Lowest energy in each k sector")
    ax1.set_xlabel(r"$k_x$")
    ax1.set_ylabel(r"$k_y$")
    ax1.set_aspect("equal", "box")
    ax1.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax1, label="Lowest sector energy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_charge_density(
    site_positions: np.ndarray,
    charge_density: np.ndarray,
    output_path: Path,
) -> None:
    # Written with Codex 02-21-26.
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sc = ax.scatter(
        site_positions[:, 0],
        site_positions[:, 1],
        c=charge_density,
        s=120.0,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_title(r"Charge density $\langle n_i \rangle$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label=r"$\langle n_i \rangle$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_correlation(
    r_values: np.ndarray,
    g_r: np.ndarray,
    output_path: Path,
) -> None:
    # Written with Codex 02-21-26.
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(r_values, g_r, "o-", color="tab:blue", lw=1.5, ms=4.0)
    ax.set_title("Pair correlation")
    ax.set_xlabel("Distance r")
    ax.set_ylabel("g(r)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_structure_factor(
    q_vectors: np.ndarray,
    s_q: np.ndarray,
    q_shell: np.ndarray,
    s_q_shell: np.ndarray,
    s_q_shell_err: np.ndarray,
    output_path: Path,
) -> None:
    # Written with Codex 02-21-26.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.4))

    sc = ax0.scatter(
        q_vectors[:, 0],
        q_vectors[:, 1],
        c=s_q,
        s=95.0,
        cmap="magma",
        edgecolors="black",
        linewidths=0.25,
    )
    ax0.set_title("Static structure factor in q-space")
    ax0.set_xlabel(r"$q_x$")
    ax0.set_ylabel(r"$q_y$")
    ax0.set_aspect("equal", "box")
    ax0.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax0, label=r"$S(\mathbf{q})$")

    ax1.errorbar(
        q_shell,
        s_q_shell,
        yerr=s_q_shell_err,
        fmt="o-",
        color="tab:orange",
        lw=1.3,
        ms=4.0,
        capsize=2,
    )
    ax1.set_title("Radial average of structure factor")
    ax1.set_xlabel(r"$|\mathbf{q}|$")
    ax1.set_ylabel(r"$S(q)$")
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
