#!/bin/bash

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 4
#SBATCH -G 1

set -euo pipefail

# Run your application
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -x "${SUBMIT_DIR}/.venv/bin/python" ]; then
  REPO_ROOT="${SUBMIT_DIR}"
elif [ -x "${SUBMIT_DIR}/../.venv/bin/python" ]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not find .venv/bin/python from submit dir: ${SUBMIT_DIR}" >&2
  exit 1
fi

echo "SCRIPT_VERSION: 2026-03-05-v3"
echo "host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "Running JAX GPU diagnostic..."
srun --ntasks=1 "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import traceback

print("diag: starting")
try:
    import jax
    import jaxlib
    print("diag: jax", jax.__version__)
    print("diag: jaxlib", jaxlib.__version__)
    print("diag: default_backend", jax.default_backend())
    print("diag: devices", jax.devices())
except Exception:
    print("diag: exception while initializing jax")
    traceback.print_exc()
PY

srun --ntasks=1 "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_haldane_psiformer.py"
