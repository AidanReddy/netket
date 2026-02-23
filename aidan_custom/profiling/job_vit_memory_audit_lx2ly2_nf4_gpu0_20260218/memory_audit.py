import json
import os
import subprocess
import time
from pathlib import Path


# Written with Codex 02-18-26.
def query_gpu_memory_mib(gpu_index: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ],
        text=True,
    ).strip()
    return int(out)


# Written with Codex 02-18-26.
def query_process_gpu_memory_mib(gpu_index: int, pid: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return 0

    if not out:
        return 0

    used = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] == str(pid):
            try:
                used += int(parts[1])
            except ValueError:
                continue
    return used


# Written with Codex 02-18-26.
def bytes_to_mib(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


# Written with Codex 02-18-26.
def main() -> None:
    # Keep defaults overridable from launcher.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    gpu_index = int(os.environ.get("AUDIT_GPU_INDEX", "0"))

    job_dir = Path("results/job_vit_memory_audit_lx2ly2_nf4_gpu0_20260218")
    job_dir.mkdir(parents=True, exist_ok=True)

    run_tag = os.environ.get("AUDIT_TAG", "default")
    driver_kind = os.environ.get("AUDIT_DRIVER", "vmc_sr").strip().lower()

    records = []
    pid = os.getpid()

    # Written with Codex 02-18-26.
    def snapshot(stage: str) -> None:
        records.append(
            {
                "stage": stage,
                "time_unix_s": time.time(),
                "gpu_memory_total_used_mib": query_gpu_memory_mib(gpu_index),
                "gpu_memory_this_pid_mib": query_process_gpu_memory_mib(gpu_index, pid),
            }
        )

    snapshot("start_before_jax_import")

    import numpy as np
    import jax

    snapshot("after_import_jax")
    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    snapshot("after_jax_device_init")

    if backend != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {backend}.")

    import netket as nk
    import optax

    from aidan_custom.haldane_model import build_haldane_hamiltonian
    from aidan_custom.models import (
        LogSlaterSpatialViT,
        make_translation_equivariant_pair_data_from_graph,
    )

    snapshot("after_import_netket_project")

    Lx, Ly = 2, 2
    n_fermions = 4
    V1 = 0.0
    t1 = 1.0
    t2 = -1 / (4 * np.cos(0.65))
    phi = 0.65
    m = 0.0

    graph, hi, ham = build_haldane_hamiltonian(
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
        n_fermions=n_fermions,
    )
    ham_sr = ham.to_fermionoperator2nd()
    snapshot("after_build_hamiltonian")

    pair_classes, pair_distances, _ = make_translation_equivariant_pair_data_from_graph(graph)
    pair_classes_hashable = tuple(tuple(int(v) for v in row) for row in pair_classes)
    pair_distances_hashable = tuple(float(v) for v in pair_distances)
    snapshot("after_pair_class_data")

    model = LogSlaterSpatialViT(
        hilbert=hi,
        num_layers=2,
        d_model=32,
        n_heads=4,
        pair_classes=pair_classes_hashable,
        pair_distances=pair_distances_hashable,
        slater_param_dtype=np.float64,
        mlp_hidden_factor=4,
        output_hidden_dim=32,
        xi_epsilon=1.0e-6,
    )
    snapshot("after_model_construct")

    vstate = nk.vqs.FullSumState(hi, model, seed=1234)
    snapshot("after_vstate_init")

    import jax.numpy as jnp

    n_params = 0
    param_bytes = 0
    for leaf in jax.tree_util.tree_leaves(vstate.parameters):
        arr = jnp.asarray(leaf)
        n_params += int(arr.size)
        param_bytes += int(arr.size) * int(arr.dtype.itemsize)

    n_states = int(vstate.hilbert.n_states)
    jac_complex128_mib = bytes_to_mib(n_states * n_params * 16)
    jac_complex64_mib = bytes_to_mib(n_states * n_params * 8)

    optimizer = nk.optimizer.Sgd(
        learning_rate=optax.linear_schedule(
            init_value=0.01,
            end_value=0.001,
            transition_steps=2,
        )
    )
    if driver_kind == "vmc_sr":
        driver = nk.driver.VMC_SR(
            ham_sr,
            optimizer,
            variational_state=vstate,
            diag_shift=0.05,
            mode="complex",
        )
    elif driver_kind == "vmc":
        driver = nk.driver.VMC(
            ham_sr,
            optimizer,
            variational_state=vstate,
        )
    else:
        raise ValueError(f"Unknown AUDIT_DRIVER={driver_kind!r}.")
    snapshot("after_driver_construct")

    # Trigger full-sum/SR compilation and one optimization update.
    driver.run(n_iter=1, out=None, show_progress=False)
    snapshot("after_driver_run_1_iter")

    # A second step after compilation often reflects steady-state memory.
    driver.run(n_iter=1, out=None, show_progress=False)
    snapshot("after_driver_run_2_iter")

    output = {
        "run_tag": run_tag,
        "driver_kind": driver_kind,
        "pid": pid,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        },
        "jax": {
            "backend": backend,
            "devices": devices,
        },
        "system": {
            "Lx": Lx,
            "Ly": Ly,
            "n_fermions": n_fermions,
            "n_states_fullsum": n_states,
        },
        "model": {
            "n_parameters": n_params,
            "parameter_storage_mib": bytes_to_mib(param_bytes),
            "jacobian_estimate_complex128_mib": jac_complex128_mib,
            "jacobian_estimate_complex64_mib": jac_complex64_mib,
        },
        "memory_timeline": records,
    }

    out_path = job_dir / f"memory_profile_{run_tag}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"backend={backend} devices={devices}")
    print(f"n_states={n_states} n_parameters={n_params}")
    print(f"parameter_storage_mib={bytes_to_mib(param_bytes):.6f}")
    print(f"jacobian_estimate_complex128_mib={jac_complex128_mib:.6f}")
    print(f"jacobian_estimate_complex64_mib={jac_complex64_mib:.6f}")
    for rec in records:
        print(
            "stage={stage} total_gpu_mib={tot} pid_gpu_mib={pidm}".format(
                stage=rec["stage"],
                tot=rec["gpu_memory_total_used_mib"],
                pidm=rec["gpu_memory_this_pid_mib"],
            )
        )


if __name__ == "__main__":
    main()
