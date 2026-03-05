#!/bin/bash

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH -c 4
#SBATCH -G h100:1

# Run your application
.venv/bin/python python ../scripts/run_haldane_psiformer.py