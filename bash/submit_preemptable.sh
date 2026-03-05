#!/bin/bash

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 4
#SBATCH -G h100:1

# Run your application
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_haldane_psiformer.py"
