from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from aidan_custom.bloch_ed import (
    build_fock_basis,
    build_many_body_hamiltonian_dense,
    build_many_body_hamiltonian_sparse,
    build_mote2_three_orbital_real_space_terms,
    build_projected_hamiltonian_terms,
    build_projected_orbital_basis,
    compute_bloch_band_data,
    mote2_site_positions_and_supercell_vectors,
    pair_correlation_from_corr_matrix,
    ordinary_density_observables_from_projected_state,
    plot_charge_density,
    plot_k_resolved_spectrum,
    plot_pair_correlation,
    plot_structure_factor,
    prepare_many_body_operator_data,
    q_vectors_from_lattice_mote2,
    radial_average,
    structure_factor_from_corr_matrix,
    write_spectrum_csv,
)


PARAMS = {
    "selected_bands": (0,),
    "n_fermions": 9,
    "supercell_matrix": np.array([[3, 6], [-3, 3]], dtype=np.int64),
    "V1": 1.0,
    "delta": 5.426542963374432,
    "ez": 0.0,
    "t_th1": 1.0,
    "t_hh1": 0.4689334070165771,
    "t_th2": -0.07151471515876148,
    "t_hh3": 0.037340187144172164,
    "t_tt1": -0.0377535889269079,
    "a_m": 1.0,
    "ph_conj": True,
}

N_EIGS_PER_SECTOR = 10
HAMILTONIAN_CUTOFF = 1.0e-12
SPARSE_THRESHOLD = 2000


_WORKER_PROJECTED_HAMILTONIAN = None
_WORKER_OPERATOR_DATA = None
_WORKER_N_PARTICLES = None
_WORKER_N_EIGS = None
_WORKER_CUTOFF = None
_WORKER_SPARSE_THRESHOLD = None


def _use_sparse(dim: int, sparse_threshold: int) -> bool:
    # Written with Codex 02-21-26.
    return bool(dim > int(sparse_threshold))


def _build_sector_basis(projected_hamiltonian, n_particles: int, sector: tuple[int, int]):
    # Written with Codex 02-21-26.
    return build_fock_basis(
        n_orbitals=int(projected_hamiltonian.one_body.shape[0]),
        n_particles=int(n_particles),
        orbital_momenta=projected_hamiltonian.orbital_momenta,
        lattice_shape=(int(projected_hamiltonian.lattice.Lx), int(projected_hamiltonian.lattice.Ly)),
        momentum_sector=sector,
    )


def _compute_sector_spectrum(
    projected_hamiltonian,
    n_particles: int,
    sector: tuple[int, int],
    n_eigs: int,
    cutoff: float,
    sparse_threshold: int,
    operator_data=None,
) -> dict[str, object]:
    # Written with Codex 02-21-26.
    basis = _build_sector_basis(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=int(n_particles),
        sector=sector,
    )
    dim = len(basis.states)
    if dim == 0:
        return {
            "sector": (int(sector[0]), int(sector[1])),
            "dimension": 0,
            "solver": "none",
            "eigenvalues": np.asarray([], dtype=np.float64),
        }

    n_eigs_eff = min(int(n_eigs), int(dim))
    use_sparse = _use_sparse(dim=dim, sparse_threshold=sparse_threshold)

    if use_sparse:
        h_sector = build_many_body_hamiltonian_sparse(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=float(cutoff),
            hermitize=True,
            operator_data=operator_data,
        )
        if n_eigs_eff < (dim - 1):
            eigvals = eigsh(
                h_sector,
                k=n_eigs_eff,
                which="SA",
                return_eigenvectors=False,
                tol=1.0e-10,
            )
            eigvals = np.sort(np.real(np.asarray(eigvals, dtype=np.float64)))
        else:
            eigvals = np.linalg.eigvalsh(h_sector.toarray())[:n_eigs_eff]
        solver_name = "sparse"
    else:
        h_sector = build_many_body_hamiltonian_dense(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=float(cutoff),
            hermitize=True,
            operator_data=operator_data,
        )
        eigvals = np.linalg.eigvalsh(h_sector)[:n_eigs_eff]
        solver_name = "dense"

    return {
        "sector": (int(sector[0]), int(sector[1])),
        "dimension": int(dim),
        "solver": solver_name,
        "eigenvalues": np.asarray(eigvals, dtype=np.float64),
    }


def _init_spectrum_worker(projected_hamiltonian, operator_data, n_particles: int, n_eigs: int, cutoff: float, sparse_threshold: int) -> None:
    # Written with Codex 02-21-26.
    global _WORKER_PROJECTED_HAMILTONIAN
    global _WORKER_OPERATOR_DATA
    global _WORKER_N_PARTICLES
    global _WORKER_N_EIGS
    global _WORKER_CUTOFF
    global _WORKER_SPARSE_THRESHOLD

    _WORKER_PROJECTED_HAMILTONIAN = projected_hamiltonian
    _WORKER_OPERATOR_DATA = operator_data
    _WORKER_N_PARTICLES = int(n_particles)
    _WORKER_N_EIGS = int(n_eigs)
    _WORKER_CUTOFF = float(cutoff)
    _WORKER_SPARSE_THRESHOLD = int(sparse_threshold)


def _solve_sector_spectrum_worker(sector: tuple[int, int]) -> dict[str, object]:
    # Written with Codex 02-21-26.
    if _WORKER_PROJECTED_HAMILTONIAN is None:
        raise RuntimeError("Worker projected Hamiltonian was not initialized.")

    return _compute_sector_spectrum(
        projected_hamiltonian=_WORKER_PROJECTED_HAMILTONIAN,
        n_particles=int(_WORKER_N_PARTICLES),
        sector=(int(sector[0]), int(sector[1])),
        n_eigs=int(_WORKER_N_EIGS),
        cutoff=float(_WORKER_CUTOFF),
        sparse_threshold=int(_WORKER_SPARSE_THRESHOLD),
        operator_data=_WORKER_OPERATOR_DATA,
    )


def _solve_ground_state_in_sector(
    projected_hamiltonian,
    n_particles: int,
    sector: tuple[int, int],
    cutoff: float,
    sparse_threshold: int,
    operator_data=None,
):
    # Written with Codex 02-21-26.
    basis = _build_sector_basis(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=int(n_particles),
        sector=sector,
    )
    dim = len(basis.states)
    if dim == 0:
        raise RuntimeError(f"Requested ground state in empty sector {sector}.")

    use_sparse = _use_sparse(dim=dim, sparse_threshold=sparse_threshold)

    if use_sparse:
        h_sector = build_many_body_hamiltonian_sparse(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=float(cutoff),
            hermitize=True,
            operator_data=operator_data,
        )
        if dim > 1:
            eigvals, eigvecs = eigsh(
                h_sector,
                k=1,
                which="SA",
                return_eigenvectors=True,
                tol=1.0e-10,
            )
            eig0 = float(np.real(eigvals[0]))
            vec0 = np.asarray(eigvecs[:, 0], dtype=np.complex128)
        else:
            eig0 = float(np.real(h_sector.toarray()[0, 0]))
            vec0 = np.asarray([1.0 + 0.0j], dtype=np.complex128)
        solver_name = "sparse"
    else:
        h_sector = build_many_body_hamiltonian_dense(
            one_body=projected_hamiltonian.one_body,
            two_body=projected_hamiltonian.two_body,
            basis=basis,
            cutoff=float(cutoff),
            hermitize=True,
            operator_data=operator_data,
        )
        eigvals, eigvecs = np.linalg.eigh(h_sector)
        eig0 = float(np.real(eigvals[0]))
        vec0 = np.asarray(eigvecs[:, 0], dtype=np.complex128)
        solver_name = "dense"

    norm = float(np.linalg.norm(vec0))
    if norm <= 0.0:
        raise RuntimeError("Ground-state eigenvector has zero norm.")
    vec0 /= norm

    return basis, eig0, vec0, solver_name, int(dim)


def _resolve_worker_count(max_tasks: int) -> int:
    # Written with Codex 02-21-26.
    # Keep a conservative default to avoid overcommitting RAM while still parallelizing sectors.
    default_workers = min(int(max_tasks), max(1, min(int(os.cpu_count() or 1), 8)))
    env_raw = os.environ.get("ED_N_WORKERS", "").strip()
    if env_raw == "":
        return default_workers

    try:
        requested = int(env_raw)
    except ValueError:
        return default_workers

    return max(1, min(int(max_tasks), requested))


def main() -> None:
    # Written with Codex 02-21-26.
    job_dir = Path(__file__).resolve().parent
    raw_data_dir = job_dir / "raw_data"
    plots_dir = job_dir / "plots"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Building shared projected basis and Hamiltonian...")
    lattice_obs, one_body_real_space, site_terms = build_mote2_three_orbital_real_space_terms(
        delta=float(PARAMS["delta"]),
        ez=float(PARAMS["ez"]),
        t_th1=float(PARAMS["t_th1"]),
        t_hh1=float(PARAMS["t_hh1"]),
        t_th2=float(PARAMS["t_th2"]),
        t_hh3=float(PARAMS["t_hh3"]),
        t_tt1=float(PARAMS["t_tt1"]),
        a_m=float(PARAMS["a_m"]),
        ph_conj=bool(PARAMS["ph_conj"]),
        V1=float(PARAMS["V1"]),
        supercell_matrix=np.asarray(PARAMS["supercell_matrix"], dtype=np.int64),
    )
    band_data = compute_bloch_band_data(
        one_body_real_space=one_body_real_space,
        lattice=lattice_obs,
    )
    projected_basis = build_projected_orbital_basis(
        one_body_real_space=one_body_real_space,
        band_data=band_data,
        selected_bands=PARAMS["selected_bands"],
    )
    projected_hamiltonian = build_projected_hamiltonian_terms(
        projected_basis=projected_basis,
        site_terms=site_terms,
    )

    lattice = projected_hamiltonian.lattice
    print(f"Lattice shape: Lx={lattice.Lx}, Ly={lattice.Ly}")
    print(f"Projected orbitals: {projected_hamiltonian.one_body.shape[0]}")

    print("Preparing many-body operator data (Numba compact arrays)...")
    operator_data = prepare_many_body_operator_data(
        one_body=projected_hamiltonian.one_body,
        two_body=projected_hamiltonian.two_body,
        cutoff=float(HAMILTONIAN_CUTOFF),
        build_numba_compact=True,
    )

    print("Precomputing momentum-sector basis cache...")
    _ = build_fock_basis(
        n_orbitals=int(projected_hamiltonian.one_body.shape[0]),
        n_particles=int(PARAMS["n_fermions"]),
        orbital_momenta=projected_hamiltonian.orbital_momenta,
        lattice_shape=(int(lattice.Lx), int(lattice.Ly)),
        momentum_sector=(0, 0),
    )

    q_vectors, q_sector_indices = q_vectors_from_lattice_mote2(
        lattice=lattice,
        a_m=float(PARAMS["a_m"]),
    )

    sectors = [(int(kx), int(ky)) for kx in range(int(lattice.Lx)) for ky in range(int(lattice.Ly))]
    n_workers = _resolve_worker_count(max_tasks=len(sectors))
    print(f"Parallel sector solves: workers={n_workers}, sectors={len(sectors)}")

    sector_results: dict[tuple[int, int], dict[str, object]] = {}

    parallel_backend = "serial"
    if n_workers == 1:
        for idx, sector in enumerate(sectors, start=1):
            result = _compute_sector_spectrum(
                projected_hamiltonian=projected_hamiltonian,
                n_particles=int(PARAMS["n_fermions"]),
                sector=sector,
                n_eigs=int(N_EIGS_PER_SECTOR),
                cutoff=float(HAMILTONIAN_CUTOFF),
                sparse_threshold=int(SPARSE_THRESHOLD),
                operator_data=operator_data,
            )
            sector_results[sector] = result
            print(
                f"  completed {idx}/{len(sectors)}: sector={sector} "
                f"dim={result['dimension']} solver={result['solver']}"
            )
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=int(n_workers),
                initializer=_init_spectrum_worker,
                initargs=(
                    projected_hamiltonian,
                    operator_data,
                    int(PARAMS["n_fermions"]),
                    int(N_EIGS_PER_SECTOR),
                    float(HAMILTONIAN_CUTOFF),
                    int(SPARSE_THRESHOLD),
                ),
            ) as executor:
                parallel_backend = "process"
                future_map = {
                    executor.submit(_solve_sector_spectrum_worker, sector): sector for sector in sectors
                }
                done = 0
                for fut in as_completed(future_map):
                    sector = future_map[fut]
                    result = fut.result()
                    sector_results[(int(sector[0]), int(sector[1]))] = result
                    done += 1
                    print(
                        f"  completed {done}/{len(sectors)}: sector={sector} "
                        f"dim={result['dimension']} solver={result['solver']}"
                    )
        except PermissionError:
            print("Process-based parallel workers unavailable; falling back to thread-based sector parallelism.")
            with ThreadPoolExecutor(max_workers=int(n_workers)) as executor:
                parallel_backend = "thread"
                future_map = {
                    executor.submit(
                        _compute_sector_spectrum,
                        projected_hamiltonian,
                        int(PARAMS["n_fermions"]),
                        sector,
                        int(N_EIGS_PER_SECTOR),
                        float(HAMILTONIAN_CUTOFF),
                        int(SPARSE_THRESHOLD),
                        operator_data,
                    ): sector
                    for sector in sectors
                }
                done = 0
                for fut in as_completed(future_map):
                    sector = future_map[fut]
                    result = fut.result()
                    sector_results[(int(sector[0]), int(sector[1]))] = result
                    done += 1
                    print(
                        f"  completed {done}/{len(sectors)}: sector={sector} "
                        f"dim={result['dimension']} solver={result['solver']}"
                    )

    spectrum_rows: list[dict[str, float]] = []
    sector_dims: dict[str, int] = {}
    sector_solvers: dict[str, str] = {}

    best_energy = np.inf
    best_sector: tuple[int, int] | None = None

    for kx in range(int(lattice.Lx)):
        for ky in range(int(lattice.Ly)):
            sector = (int(kx), int(ky))
            result = sector_results[sector]
            dim = int(result["dimension"])
            solver_name = str(result["solver"])
            eigvals = np.asarray(result["eigenvalues"], dtype=np.float64)

            key = f"{kx},{ky}"
            sector_dims[key] = dim
            sector_solvers[key] = solver_name

            q_idx = int(kx) * int(lattice.Ly) + int(ky)
            q_vec = q_vectors[q_idx]
            for eig_idx, eig in enumerate(eigvals):
                spectrum_rows.append(
                    {
                        "sector_kx": int(kx),
                        "sector_ky": int(ky),
                        "kvec_x": float(q_vec[0]),
                        "kvec_y": float(q_vec[1]),
                        "eig_index": int(eig_idx),
                        "energy": float(np.real(eig)),
                        "sector_dimension": int(dim),
                        "solver": solver_name,
                    }
                )

            if eigvals.size > 0 and float(eigvals[0]) < best_energy:
                best_energy = float(eigvals[0])
                best_sector = sector

    if best_sector is None:
        raise RuntimeError("No non-empty momentum sector found.")

    print(f"Global ground state sector from spectrum scan: {best_sector}, energy={best_energy:.12f}")

    best_basis, ground_energy_exact, ground_vec, ground_solver, ground_dim = _solve_ground_state_in_sector(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=int(PARAMS["n_fermions"]),
        sector=best_sector,
        cutoff=float(HAMILTONIAN_CUTOFF),
        sparse_threshold=int(SPARSE_THRESHOLD),
        operator_data=operator_data,
    )
    print(
        f"Solved ground-state eigenvector in sector {best_sector}: "
        f"dim={ground_dim} solver={ground_solver} energy={ground_energy_exact:.12f}"
    )

    # observables = ordinary_density_observables_from_projected_state(
    #     basis=best_basis,
    #     coefficients=ground_vec,
    #     site_amplitudes=projected_basis.site_amplitudes,
    # )
    # charge_density = np.asarray(observables["charge_density"], dtype=np.float64)
    # corr_matrix = np.asarray(observables["corr_matrix"], dtype=np.complex128)

    # site_positions, supercell_t1, supercell_t2 = mote2_site_positions_and_supercell_vectors(
    #     lattice=lattice_obs,
    #     a_m=float(PARAMS["a_m"]),
    # )
    # r_values, g_r = pair_correlation_from_corr_matrix(
    #     corr_matrix=corr_matrix,
    #     site_positions=site_positions,
    #     n_fermions=int(PARAMS["n_fermions"]),
    #     supercell_t1=supercell_t1,
    #     supercell_t2=supercell_t2,
    # )
    # s_q = structure_factor_from_corr_matrix(
    #     corr_matrix=corr_matrix,
    #     site_positions=site_positions,
    #     q_vectors=q_vectors,
    #     n_fermions=int(PARAMS["n_fermions"]),
    # )
    # q_shell, s_q_shell, s_q_shell_err = radial_average(
    #     values=s_q,
    #     vectors=q_vectors,
    #     zero_mode_to_zero=True,
    # )

    write_spectrum_csv(raw_data_dir / "k_sector_spectrum.csv", spectrum_rows)

    np.savez(
        raw_data_dir / "k_sector_spectrum.npz",
        sector_kx=np.asarray([int(row["sector_kx"]) for row in spectrum_rows], dtype=np.int64),
        sector_ky=np.asarray([int(row["sector_ky"]) for row in spectrum_rows], dtype=np.int64),
        kvec_x=np.asarray([float(row["kvec_x"]) for row in spectrum_rows], dtype=np.float64),
        kvec_y=np.asarray([float(row["kvec_y"]) for row in spectrum_rows], dtype=np.float64),
        eig_index=np.asarray([int(row["eig_index"]) for row in spectrum_rows], dtype=np.int64),
        energy=np.asarray([float(row["energy"]) for row in spectrum_rows], dtype=np.float64),
    )

    # np.savez(
    #     raw_data_dir / "ground_state_observables.npz",
    #     ground_sector=np.asarray(best_sector, dtype=np.int64),
    #     ground_energy=np.asarray(float(ground_energy_exact), dtype=np.float64),
    #     charge_density=charge_density,
    #     corr_matrix=corr_matrix,
    #     site_positions=site_positions,
    #     pair_r_values=r_values,
    #     pair_g_r=g_r,
    #     q_vectors=q_vectors,
    #     q_sector_indices=q_sector_indices,
    #     s_q=s_q,
    #     q_shell=q_shell,
    #     s_q_shell=s_q_shell,
    #     s_q_shell_err=s_q_shell_err,
    #     supercell_t1=supercell_t1,
    #     supercell_t2=supercell_t2,
    # )

    plot_k_resolved_spectrum(
        spectrum_rows=spectrum_rows,
        lattice_shape=(int(lattice.Lx), int(lattice.Ly)),
        output_path=plots_dir / "k_resolved_energy_spectrum.png",
    )
    # plot_charge_density(
    #     site_positions=site_positions,
    #     charge_density=charge_density,
    #     output_path=plots_dir / "charge_density.png",
    # )
    # plot_pair_correlation(
    #     r_values=r_values,
    #     g_r=g_r,
    #     output_path=plots_dir / "pair_correlation.png",
    # )
    # plot_structure_factor(
    #     q_vectors=q_vectors,
    #     s_q=s_q,
    #     q_shell=q_shell,
    #     s_q_shell=s_q_shell,
    #     s_q_shell_err=s_q_shell_err,
    #     output_path=plots_dir / "structure_factor.png",
    # )

    shutil.copy2(Path(__file__).resolve(), plots_dir / "run_script_used.py")

    summary = {
        "parameters": {
            "selected_bands": [int(x) for x in PARAMS["selected_bands"]],
            "n_fermions": int(PARAMS["n_fermions"]),
            "supercell_matrix": np.asarray(PARAMS["supercell_matrix"], dtype=np.int64).tolist(),
            "V1": float(PARAMS["V1"]),
            "delta": float(PARAMS["delta"]),
            "ez": float(PARAMS["ez"]),
            "t_th1": float(PARAMS["t_th1"]),
            "t_hh1": float(PARAMS["t_hh1"]),
            "t_th2": float(PARAMS["t_th2"]),
            "t_hh3": float(PARAMS["t_hh3"]),
            "t_tt1": float(PARAMS["t_tt1"]),
            "a_m": float(PARAMS["a_m"]),
            "ph_conj": bool(PARAMS["ph_conj"]),
        },
        "projected_lattice_shape": {
            "Lx": int(lattice.Lx),
            "Ly": int(lattice.Ly),
            "n_orbitals_per_cell": int(lattice.n_orbitals_per_cell),
        },
        "projected_n_orbitals": int(projected_hamiltonian.one_body.shape[0]),
        "n_eigs_per_sector_requested": int(N_EIGS_PER_SECTOR),
        "n_workers": int(n_workers),
        "parallel_backend": str(parallel_backend),
        "ground_sector": [int(best_sector[0]), int(best_sector[1])],
        "ground_energy": float(ground_energy_exact),
        "ground_solver": str(ground_solver),
        "ground_sector_dimension": int(ground_dim),
        "sector_dimensions": sector_dims,
        "sector_solvers": sector_solvers,
        "files": {
            "spectrum_csv": str(raw_data_dir / "k_sector_spectrum.csv"),
            "spectrum_npz": str(raw_data_dir / "k_sector_spectrum.npz"),
            "observables_npz": str(raw_data_dir / "ground_state_observables.npz"),
            "k_spectrum_plot": str(plots_dir / "k_resolved_energy_spectrum.png"),
            # "charge_density_plot": str(plots_dir / "charge_density.png"),
            # "pair_correlation_plot": str(plots_dir / "pair_correlation.png"),
            # "structure_factor_plot": str(plots_dir / "structure_factor.png"),
            "run_script_used": str(plots_dir / "run_script_used.py"),
        },
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Done. Outputs saved under:")
    print(f"  {job_dir}")


if __name__ == "__main__":
    main()
