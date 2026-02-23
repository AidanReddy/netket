from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import jax
from scipy.sparse.linalg import eigsh

from .bandstructure import eigenvalues_at_k
from .geometry import PRIMITIVE_A1, PRIMITIVE_A2, reciprocal_vectors


def _read_nonnegative_int_env(var_name: str, default: int) -> int:
    # Written with Codex 02-21-26.
    raw = os.environ.get(var_name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = int(default)
    return max(value, 0)


_AIDAN_LOG_MEMORY_EVERY = _read_nonnegative_int_env("AIDAN_LOG_MEMORY_EVERY", 1)


def _append_device_memory_stats(log_data: dict) -> None:
    # Written with Codex 02-21-26.
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return
    if not stats:
        return

    key_map = {
        "bytes_in_use": "DeviceBytesInUse",
        "peak_bytes_in_use": "DevicePeakBytesInUse",
        "pool_bytes": "DevicePoolBytes",
        "peak_pool_bytes": "DevicePeakPoolBytes",
        "bytes_limit": "DeviceBytesLimit",
        "num_allocs": "DeviceNumAllocs",
    }
    for stat_key, log_key in key_map.items():
        value = stats.get(stat_key, None)
        if value is None:
            continue
        log_data[log_key] = float(value)

    for bytes_key, gib_key in (
        ("bytes_in_use", "DeviceBytesInUseGiB"),
        ("peak_bytes_in_use", "DevicePeakBytesInUseGiB"),
        ("pool_bytes", "DevicePoolGiB"),
        ("peak_pool_bytes", "DevicePeakPoolGiB"),
        ("bytes_limit", "DeviceBytesLimitGiB"),
    ):
        value = stats.get(bytes_key, None)
        if value is None:
            continue
        log_data[gib_key] = float(value) / (1024.0**3)


def exact_manybody_ground_state_energy(hamiltonian, sparse_threshold: int = 2000) -> float:
    # Written with Codex 02-18-26.
    dim = int(hamiltonian.hilbert.n_states)
    sparse_h = hamiltonian.to_sparse()
    if dim > sparse_threshold:
        eigvals = eigsh(
            sparse_h,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1e-10,
        )
        return float(np.real(eigvals[0]))

    dense_h = np.asarray(sparse_h.toarray())
    return float(np.linalg.eigvalsh(dense_h)[0].real)


def exact_noninteracting_ground_state_energy_bloch(
    Lx: int,
    Ly: int,
    n_fermions: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
) -> float:
    # Written with Codex 02-18-26.
    b1, b2 = reciprocal_vectors(PRIMITIVE_A1, PRIMITIVE_A2)
    one_body_energies = []
    for n1 in range(Lx):
        for n2 in range(Ly):
            kvec = (n1 / Lx) * b1 + (n2 / Ly) * b2
            one_body_energies.extend(
                eigenvalues_at_k(kvec[0], kvec[1], t1=t1, t2=t2, phi=phi, m=m)
            )

    one_body_energies = np.sort(np.asarray(one_body_energies, dtype=float))
    if n_fermions < 0 or n_fermions > one_body_energies.size:
        raise ValueError("n_fermions is outside the available one-body state count.")
    return float(np.sum(one_body_energies[:n_fermions]))


def exact_reference_ground_state_energy(
    hamiltonian,
    Lx: int,
    Ly: int,
    n_fermions: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    V1: float,
    sparse_threshold: int = 2000,
):
    # Written with Codex 02-18-26.
    if np.isclose(V1, 0.0):
        e_ref = exact_noninteracting_ground_state_energy_bloch(
            Lx=Lx,
            Ly=Ly,
            n_fermions=n_fermions,
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
        )
        return e_ref, "Exact GS (Bloch filling)", "Bloch-filling (non-interacting)"

    e_ref = exact_manybody_ground_state_energy(
        hamiltonian,
        sparse_threshold=sparse_threshold,
    )
    return e_ref, "Exact many-body GS (ED)", "ED (interacting)"


def exact_projected_reference_ground_state_energy(
    Lx: int,
    Ly: int,
    n_fermions: int,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    V1: float,
    selected_bands: Sequence[int] = (0,),
):
    # Written with Codex 02-20-26.
    from .bloch_ed import (
        build_haldane_projected_hamiltonian,
        solve_projected_all_momentum_sectors,
    )

    bands = tuple(int(b) for b in selected_bands)
    if len(bands) == 0:
        raise ValueError("selected_bands must contain at least one band index.")

    projected_hamiltonian = build_haldane_projected_hamiltonian(
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        selected_bands=bands,
        V1=V1,
    )
    sector_results = solve_projected_all_momentum_sectors(
        projected_hamiltonian=projected_hamiltonian,
        n_particles=n_fermions,
        n_eigs=1,
        use_sparse=True,
    )

    e0 = np.inf
    e0_sector = None
    for sector, sector_data in sector_results.items():
        eigvals = np.asarray(sector_data["eigenvalues"])
        if eigvals.size == 0:
            continue
        e_sector = float(np.real(eigvals[0]))
        if e_sector < e0:
            e0 = e_sector
            e0_sector = (int(sector[0]), int(sector[1]))

    if e0_sector is None:
        raise RuntimeError("Projected ED returned no non-empty momentum sectors.")

    if bands == (0,):
        label = "1 Band ED"
        method = (
            "ED in projected lowest Bloch band "
            f"(sector={e0_sector})"
        )
    else:
        label = f"{len(bands)} Band ED"
        method = (
            f"ED in projected Bloch bands {list(bands)} "
            f"(sector={e0_sector})"
        )

    return float(e0), label, method, e0_sector


def tree_l2_norm(tree) -> float:
    # Written with Codex 02-18-26.
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return np.nan

    norm2 = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf)
        norm2 += float(np.vdot(arr.ravel(), arr.ravel()).real)
    return float(np.sqrt(norm2))


def log_optimization_diagnostics(step: int, log_data: dict, driver) -> bool:
    # Written with Codex 02-18-26.
    dp = getattr(driver, "_dp", None)
    if dp is not None:
        log_data["UpdateNormL2"] = tree_l2_norm(dp)
    if _AIDAN_LOG_MEMORY_EVERY > 0 and int(step) % _AIDAN_LOG_MEMORY_EVERY == 0:
        _append_device_memory_stats(log_data)
    return True


def make_optimization_callback(job_dir):
    """Returns a callback that logs optimization diagnostics and GPU memory to job_dir/gpu_mem.log."""
    gpu_mem_log_path = Path(job_dir) / "gpu_mem.log"
    _log_file = open(gpu_mem_log_path, "a")  # noqa: SIM115
    _log_file.write("step\tbytes_in_use_GiB\tpeak_bytes_in_use_GiB\tbytes_limit_GiB\n")
    _log_file.flush()

    def _callback(step: int, log_data: dict, driver) -> bool:
        dp = getattr(driver, "_dp", None)
        if dp is not None:
            log_data["UpdateNormL2"] = tree_l2_norm(dp)
        if _AIDAN_LOG_MEMORY_EVERY > 0 and int(step) % _AIDAN_LOG_MEMORY_EVERY == 0:
            _append_device_memory_stats(log_data)
            in_use_gib = log_data.get("DeviceBytesInUseGiB", None)
            if in_use_gib is not None:
                peak_gib = log_data.get("DevicePeakBytesInUseGiB", float("nan"))
                limit_gib = log_data.get("DeviceBytesLimitGiB", float("nan"))
                _log_file.write(f"{step}\t{in_use_gib:.4f}\t{peak_gib:.4f}\t{limit_gib:.4f}\n")
                _log_file.flush()
        return True

    return _callback


def _series_to_float_array(values) -> np.ndarray:
    # Written with Codex 02-18-26.
    if values is None:
        return np.asarray([], dtype=float)

    if isinstance(values, np.ndarray):
        arr = np.real(values) if np.iscomplexobj(values) else values
        return np.asarray(arr, dtype=float)

    if isinstance(values, (list, tuple)):
        out = np.empty(len(values), dtype=float)
        for i, v in enumerate(values):
            if v is None:
                out[i] = np.nan
            else:
                out[i] = float(np.real(np.asarray(v)))
        return out

    return np.asarray([float(np.real(np.asarray(values)))], dtype=float)


def _series_to_int_array(values) -> np.ndarray:
    # Written with Codex 02-18-26.
    if values is None:
        return np.asarray([], dtype=int)
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=int)
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=int)
    return np.asarray([int(values)], dtype=int)


def _extract_scalar_history(log_data: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    # Written with Codex 02-18-26.
    entry = log_data.get(key, None)
    if entry is None:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    if isinstance(entry, dict):
        iters = _series_to_int_array(entry.get("iters", None))

        value = None
        if "value" in entry:
            value = entry["value"]
        elif "values" in entry:
            value = entry["values"]
        elif "Mean" in entry:
            value = entry["Mean"]
        else:
            remaining = [k for k in entry if k != "iters"]
            if len(remaining) == 1:
                value = entry[remaining[0]]

        if isinstance(value, dict):
            if "real" in value:
                value = value["real"]
            elif "imag" in value:
                value = value["imag"]
            else:
                value = []

        values = _series_to_float_array(value)
        if iters.size == 0 and values.size > 0:
            iters = np.arange(values.size, dtype=int)
        n = min(iters.size, values.size)
        return iters[:n], values[:n]

    values = _series_to_float_array(entry)
    iters = np.arange(values.size, dtype=int)
    return iters, values


def save_optimization_plots_from_jsonlog(
    job_dir: str | Path,
    train_log_path: str | Path,
    reference_energy: float | None = None,
    reference_label: str | None = None,
) -> dict[str, str]:
    # Written with Codex 02-18-26.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    job_dir = Path(job_dir)
    train_log_path = Path(train_log_path)
    plot_dir = job_dir / "plots_optimization"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with train_log_path.open("r", encoding="utf-8") as f:
        log_data = json.load(f)

    energy_entry = log_data.get("Energy", {})
    iters = _series_to_int_array(energy_entry.get("iters", []))

    mean_entry = energy_entry.get("Mean", [])
    if isinstance(mean_entry, dict) and "real" in mean_entry:
        energy_mean_real = _series_to_float_array(mean_entry.get("real", []))
    else:
        energy_mean_real = _series_to_float_array(mean_entry)

    energy_variance = _series_to_float_array(energy_entry.get("Variance", []))
    energy_tau = _series_to_float_array(energy_entry.get("TauCorr", []))
    energy_rhat = _series_to_float_array(energy_entry.get("R_hat", []))

    n_energy = min(iters.size, energy_mean_real.size, energy_variance.size)
    iters = iters[:n_energy]
    energy_mean_real = energy_mean_real[:n_energy]
    energy_variance = energy_variance[:n_energy]
    energy_std_local = np.sqrt(np.maximum(energy_variance, 0.0))

    acceptance_iters, acceptance_values = _extract_scalar_history(log_data, "acceptance")

    n_tau = min(iters.size, energy_tau.size)
    n_rhat = min(iters.size, energy_rhat.size)
    iters_tau = iters[:n_tau]
    iters_rhat = iters[:n_rhat]
    energy_tau = energy_tau[:n_tau]
    energy_rhat = energy_rhat[:n_rhat]

    np.savez(
        plot_dir / "optimization_series.npz",
        iters=iters,
        energy_mean_real=energy_mean_real,
        energy_std_local=energy_std_local,
        acceptance_iters=acceptance_iters,
        acceptance_values=acceptance_values,
        taucorr_iters=iters_tau,
        taucorr_values=energy_tau,
        rhat_iters=iters_rhat,
        rhat_values=energy_rhat,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, energy_mean_real, lw=1.8, color="tab:blue", label="VMC_SR")
    if reference_energy is not None:
        label = reference_label if reference_label is not None else "Reference energy"
        ax.axhline(reference_energy, color="black", ls="--", lw=1.5, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs optimization")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    energy_plot_path = plot_dir / "energy_vs_optimization.png"
    fig.savefig(energy_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, energy_std_local, color="tab:purple", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\sigma(E_\mathrm{loc})$")
    ax.set_title("Energy standard deviation")
    ax.grid(alpha=0.3)
    energy_std_plot_path = plot_dir / "energy_std_vs_optimization.png"
    fig.savefig(energy_std_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if acceptance_values.size > 0:
        ax.plot(acceptance_iters, acceptance_values, color="tab:green", lw=1.5)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "Acceptance unavailable (e.g. FullSumState)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance")
    ax.set_title("MC acceptance ratio")
    ax.grid(alpha=0.3)
    acceptance_plot_path = plot_dir / "acceptance_ratio.png"
    fig.savefig(acceptance_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    has_diag = False
    if energy_tau.size > 0 and np.any(np.isfinite(energy_tau)):
        ax.plot(iters_tau, energy_tau, color="tab:orange", lw=1.5, label="TauCorr")
        has_diag = True
    if energy_rhat.size > 0 and np.any(np.isfinite(energy_rhat)):
        ax.plot(iters_rhat, energy_rhat, color="tab:brown", lw=1.5, label="R_hat")
        has_diag = True
    if has_diag:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "No sampling diagnostics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Diagnostic value")
    ax.set_title("Sampling diagnostics")
    ax.grid(alpha=0.3)
    diagnostics_plot_path = plot_dir / "sampling_diagnostics.png"
    fig.savefig(diagnostics_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_paths = {
        "energy_plot": str(energy_plot_path),
        "energy_std_plot": str(energy_std_plot_path),
        "acceptance_plot": str(acceptance_plot_path),
        "sampling_diagnostics_plot": str(diagnostics_plot_path),
        "raw_series": str(plot_dir / "optimization_series.npz"),
    }

    with (plot_dir / "plot_files.json").open("w", encoding="utf-8") as f:
        json.dump(output_paths, f, indent=2)

    return output_paths
