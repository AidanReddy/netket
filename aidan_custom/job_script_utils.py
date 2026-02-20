from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_HISTORY_KEYS = (
    "iters",
    "energy_mean",
    "energy_sigma",
    "energy_variance",
    "energy_std_local",
    "energy_tau",
    "energy_rhat",
    "update_norm_iters",
    "update_norm_values",
)


def build_resume_setup_signature(
    *,
    Lx: int,
    Ly: int,
    n_fermions: int,
    V1: float,
    t1: float,
    t2: float,
    phi: float,
    m: float,
    model_type: str,
    sample_type: str,
    vit_num_layers: int,
    vit_d_model: int,
    vit_n_heads: int,
    vit_mlp_hidden_factor: int,
    vit_output_hidden_dim: int,
    vit_xi_epsilon: float,
    slater_init_mode: str,
    n_samples: int,
    n_discard_per_chain: int,
    sweep_size: int,
    n_chains: int,
    optimizer_name: str,
    diag_shift: float,
    mode: str,
) -> dict[str, object]:
    # Written with Codex 02-19-26.
    return {
        "system": {
            "Lx": int(Lx),
            "Ly": int(Ly),
            "n_fermions": int(n_fermions),
            "V1": float(V1),
            "t1": float(t1),
            "t2": float(t2),
            "phi": float(phi),
            "m": float(m),
        },
        "network": {
            "model_type": str(model_type),
            "sample_type": str(sample_type),
            "vit_num_layers": int(vit_num_layers),
            "vit_d_model": int(vit_d_model),
            "vit_n_heads": int(vit_n_heads),
            "vit_mlp_hidden_factor": int(vit_mlp_hidden_factor),
            "vit_output_hidden_dim": int(vit_output_hidden_dim),
            "vit_xi_epsilon": float(vit_xi_epsilon),
            "slater_init_mode": str(slater_init_mode),
        },
        "optimization": {
            "n_samples": int(n_samples),
            "n_discard_per_chain": int(n_discard_per_chain),
            "sweep_size": int(sweep_size),
            "n_chains": int(n_chains),
            "optimizer": str(optimizer_name),
            "diag_shift": float(diag_shift),
            "mode": str(mode),
        },
    }


def _fallback_resume_setup_from_summary(summary: dict[str, object]) -> dict[str, object]:
    # Written with Codex 02-19-26.
    system = summary.get("system", {})
    network = summary.get("network", {})
    optimization = summary.get("optimization", {})

    if not isinstance(system, dict):
        system = {}
    if not isinstance(network, dict):
        network = {}
    if not isinstance(optimization, dict):
        optimization = {}

    return {
        "system": {
            "Lx": system.get("Lx"),
            "Ly": system.get("Ly"),
            "n_fermions": system.get("n_fermions"),
            "V1": system.get("V1"),
            "t1": system.get("t1"),
            "t2": system.get("t2"),
            "phi": system.get("phi"),
            "m": system.get("m"),
        },
        "network": {
            "model_type": network.get("model_type"),
            "sample_type": network.get("sample_type"),
            "vit_num_layers": network.get("vit_num_layers"),
            "vit_d_model": network.get("vit_d_model"),
            "vit_n_heads": network.get("vit_n_heads"),
            "vit_mlp_hidden_factor": network.get("vit_mlp_hidden_factor"),
            "vit_output_hidden_dim": network.get("vit_output_hidden_dim"),
            "vit_xi_epsilon": network.get("vit_xi_epsilon"),
            "slater_init_mode": network.get("slater_init_mode"),
        },
        "optimization": {
            "n_samples": optimization.get("n_samples"),
            "n_discard_per_chain": optimization.get("n_discard_per_chain"),
            "sweep_size": optimization.get("sweep_size"),
            "n_chains": optimization.get("n_chains"),
            "optimizer": optimization.get("optimizer"),
            "diag_shift": optimization.get("diag_shift"),
            "mode": optimization.get("mode"),
        },
    }


def _setup_from_summary(summary: dict[str, object]) -> dict[str, object]:
    # Written with Codex 02-19-26.
    setup = summary.get("resume_setup")
    if isinstance(setup, dict):
        return setup
    return _fallback_resume_setup_from_summary(summary)


def find_latest_matching_resume_job(
    *,
    results_dir: Path,
    resume_setup_signature: dict[str, object],
    exclude_job_dir: Path,
) -> tuple[Path | None, dict[str, object] | None]:
    # Written with Codex 02-19-26.
    best_job_dir: Path | None = None
    best_summary: dict[str, object] | None = None
    best_mtime = -1.0

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() == exclude_job_dir.resolve():
            continue

        summary_path = child / "summary.json"
        ckpt_path = child / "vstate_variables.mpack"
        if not summary_path.exists() or not ckpt_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue

        if _setup_from_summary(summary) != resume_setup_signature:
            continue

        mtime = summary_path.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best_job_dir = child
            best_summary = summary

    return best_job_dir, best_summary


def find_resume_source_for_setup(
    *,
    job_dir: Path,
    results_dir: Path,
    resume_setup_signature: dict[str, object],
) -> tuple[Path | None, dict[str, object] | None]:
    # Written with Codex 02-19-26.
    local_summary_path = job_dir / "summary.json"
    local_ckpt_path = job_dir / "vstate_variables.mpack"
    if local_summary_path.exists() and local_ckpt_path.exists():
        try:
            local_summary = json.loads(local_summary_path.read_text())
        except Exception:
            local_summary = None
        if isinstance(local_summary, dict):
            if _setup_from_summary(local_summary) == resume_setup_signature:
                return job_dir, local_summary

    return find_latest_matching_resume_job(
        results_dir=results_dir,
        resume_setup_signature=resume_setup_signature,
        exclude_job_dir=job_dir,
    )


def _empty_history() -> dict[str, np.ndarray]:
    # Written with Codex 02-19-26.
    return {
        "iters": np.asarray([], dtype=np.int64),
        "energy_mean": np.asarray([], dtype=np.complex128),
        "energy_sigma": np.asarray([], dtype=np.float64),
        "energy_variance": np.asarray([], dtype=np.float64),
        "energy_std_local": np.asarray([], dtype=np.float64),
        "energy_tau": np.asarray([], dtype=np.float64),
        "energy_rhat": np.asarray([], dtype=np.float64),
        "update_norm_iters": np.asarray([], dtype=np.int64),
        "update_norm_values": np.asarray([], dtype=np.float64),
    }


def load_history_for_append(job_dir: Path) -> dict[str, np.ndarray] | None:
    # Written with Codex 02-19-26.
    contiguous_path = job_dir / "raw_data" / "optimization_history_contiguous.npz"
    latest_path = job_dir / "raw_data" / "optimization_history.npz"

    source_path: Path | None = None
    if contiguous_path.exists():
        source_path = contiguous_path
    elif latest_path.exists():
        source_path = latest_path

    if source_path is None:
        return None

    loaded: dict[str, np.ndarray] = _empty_history()
    with np.load(source_path, allow_pickle=False) as data:
        for key in _HISTORY_KEYS:
            if key in data:
                loaded[key] = np.asarray(data[key])
    return loaded


def concatenate_histories(
    previous: dict[str, np.ndarray] | None,
    current: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    # Written with Codex 02-19-26.
    if previous is None or previous["iters"].size == 0:
        return {key: np.asarray(value) for key, value in current.items()}

    out = _empty_history()
    offset = int(previous["iters"][-1]) + 1

    out["iters"] = np.concatenate([previous["iters"], np.asarray(current["iters"]) + offset])
    out["energy_mean"] = np.concatenate([previous["energy_mean"], np.asarray(current["energy_mean"])])
    out["energy_sigma"] = np.concatenate([previous["energy_sigma"], np.asarray(current["energy_sigma"])])
    out["energy_variance"] = np.concatenate([previous["energy_variance"], np.asarray(current["energy_variance"])])
    out["energy_std_local"] = np.concatenate([previous["energy_std_local"], np.asarray(current["energy_std_local"])])
    out["energy_tau"] = np.concatenate([previous["energy_tau"], np.asarray(current["energy_tau"])])
    out["energy_rhat"] = np.concatenate([previous["energy_rhat"], np.asarray(current["energy_rhat"])])

    if np.asarray(current["update_norm_iters"]).size > 0:
        shifted_update_iters = np.asarray(current["update_norm_iters"]) + offset
    else:
        shifted_update_iters = np.asarray(current["update_norm_iters"])

    out["update_norm_iters"] = np.concatenate([previous["update_norm_iters"], shifted_update_iters])
    out["update_norm_values"] = np.concatenate([previous["update_norm_values"], np.asarray(current["update_norm_values"])])
    return out


def save_history_npz(path: Path, history: dict[str, np.ndarray]) -> None:
    # Written with Codex 02-19-26.
    np.savez(path, **history)
