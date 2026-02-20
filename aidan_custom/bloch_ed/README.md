# Bloch-Space Exact Diagonalization

This package implements exact diagonalization in a Bloch-state basis with:

- crystal-momentum-resolved many-body sectors;
- projection to user-selected Bloch bands;
- separate stages for:
  - single-particle Bloch/band construction;
  - projected matrix-element construction;
  - many-body Hamiltonian assembly and diagonalization.

## Core workflow

1. Build a lattice embedding and one-body real-space matrix.
2. Diagonalize one-body Bloch blocks: `compute_bloch_band_data`.
3. Build selected-band orbitals: `build_projected_orbital_basis`.
4. Build projected interactions: `build_projected_hamiltonian_terms`.
5. Build/solve momentum-sector ED:
   - `solve_projected_sector`
   - `solve_projected_all_momentum_sectors`

## Operator convention for quartic terms

Quartic terms are represented as:

`coefficient * c†_a c†_b c_d c_c`

The stored index order is `(create_1, create_2, annihilate_1, annihilate_2) = (a, b, c, d)`.

For density-density interactions with `i != j`:

`V * n_i n_j = V * c†_i c†_j c_j c_i`

so each term is encoded as `(i, j, i, j, V)`.

## Haldane helper entry points

- `build_haldane_real_space_terms`
- `build_haldane_projected_hamiltonian`

See `example_haldane_ed.py` for a minimal runnable example.
