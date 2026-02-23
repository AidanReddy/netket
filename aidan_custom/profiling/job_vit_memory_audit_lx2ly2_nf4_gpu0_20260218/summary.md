# GPU Memory Audit Summary (`Lx=Ly=2`, `Nf=4`, `FullSumState`)

## Files
- `results/job_vit_memory_audit_lx2ly2_nf4_gpu0_20260218/memory_audit.py`
- `results/job_vit_memory_audit_lx2ly2_nf4_gpu0_20260218/memory_profile_default_alloc.json`
- `results/job_vit_memory_audit_lx2ly2_nf4_gpu0_20260218/memory_profile_prealloc_false.json`
- `results/job_vit_memory_audit_lx2ly2_nf4_gpu0_20260218/memory_profile_prealloc_false_vmc.json`

## Key Results (this process only, MiB)
- Default allocator (`VMC_SR`): jumps to ~36914 MiB at NetKet import (JAX preallocation), then ~36974 MiB after first SR step.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (`VMC_SR`): stays ~472 MiB before optimization, then jumps to ~33390 MiB at first SR step.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (`VMC`, no SR): stays ~472 MiB before optimization, then only ~586 MiB after first step.

## Interpretation
- Most of the extreme memory usage comes from **SR/QGT construction and solve**, not from model parameters, Hamiltonian build, or FullSum itself.
- Parameter storage is tiny: ~0.18 MiB (`23704` parameters).
- FullSum Jacobian estimate is modest:
  - complex128: ~25.32 MiB
  - complex64: ~12.66 MiB
- But dense SR matrix scales as `P x P` with `P=23704`:
  - `P^2 = 561,879,616` entries
  - one complex128 dense matrix is ~8.37 GiB
  - multiple matrices/workspaces during linear solve can easily reach tens of GiB, consistent with ~33 GiB observed.
