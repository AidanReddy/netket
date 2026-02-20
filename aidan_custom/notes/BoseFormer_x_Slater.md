# BoseFormer x Slater Architecture Plan

This note defines a concrete architecture plan for a new ansatz,

$$
\log \Psi_{\text{total}}(n) = \log \Psi_{\text{Slater}}(n) + \log \Psi_{\text{BoseFormer}}(n),
$$

where the BoseFormer factor is a Transformer over occupied-particle coordinates using periodified Cartesian inputs on a lattice with PBC.

## Key Distinction From SpatialViT

- `SpatialViT`: token set is all lattice sites, and each token starts from occupation $n_i\in\{0,1\}$.
- `BoseFormer` (this plan): token set is occupied particles only, and each token starts from coordinate-derived periodic features of $\mathbf{r}_{i_p}$.
- Occupations are used only to extract occupied indices $I(n)$; they are not used as token features.

## 0) Notation

- Configuration: $n \in \{0,1\}^{N_s}$, with fixed particle number $N_f$.
- Occupied indices (sorted): $I(n) = (i_1,\dots,i_{N_f})$, where $n_{i_p}=1$.
- Site positions: $\mathbf{r}_i \in \mathbb{R}^2$, $i=1,\dots,N_s$.
- Primitive lattice vectors: $(\mathbf{a}_1,\mathbf{a}_2)$, supercell extents $(L_x,L_y)$.
- Primitive reciprocal vectors $(\mathbf{b}_1,\mathbf{b}_2)$ satisfy
  \[
  \mathbf{b}_\mu \cdot \mathbf{a}_\nu = 2\pi \delta_{\mu\nu}.
  \]
- Supercell reciprocal vectors:
  \[
  \mathbf{G}_1 = \frac{\mathbf{b}_1}{L_x}, \qquad \mathbf{G}_2 = \frac{\mathbf{b}_2}{L_y}.
  \]

## 1) Occupied-Particle Tokens From Lattice Configurations

From $n$, build the occupied coordinate list

$$
\mathbf{x}_p \equiv \mathbf{r}_{i_p}, \quad p=1,\dots,N_f.
$$

Implementation detail for fixed $N_f$:

$$
I(n)=\texttt{nonzero}(n,\ \texttt{size}=N_f),
$$

then gather $\mathbf{r}_{I(n)}$. This avoids variable-length control flow and stays JIT-friendly.

## 2) Periodified Cartesian Input Features

For each occupied particle $p$, define the base feature

$$
\mathbf{u}_p =
\begin{bmatrix}
\cos(\mathbf{x}_p\!\cdot\!\mathbf{G}_1) \\
\sin(\mathbf{x}_p\!\cdot\!\mathbf{G}_1) \\
\cos(\mathbf{x}_p\!\cdot\!\mathbf{G}_2) \\
\sin(\mathbf{x}_p\!\cdot\!\mathbf{G}_2)
\end{bmatrix}
\in \mathbb{R}^4.
$$

Optional harmonic enrichment (if needed):

$$
\mathbf{u}^{(m)}_p =
\big[\cos(m\,\mathbf{x}_p\!\cdot\!\mathbf{G}_1),\ \sin(m\,\mathbf{x}_p\!\cdot\!\mathbf{G}_1),\ \cos(m\,\mathbf{x}_p\!\cdot\!\mathbf{G}_2),\ \sin(m\,\mathbf{x}_p\!\cdot\!\mathbf{G}_2)\big],
$$

concatenated for $m=1,\dots,M_h$.

Then project to model dimension $d$:

$$
\mathbf{h}^{(0)}_p = W_{\text{in}} \mathbf{u}_p + \mathbf{b}_{\text{in}}, \qquad W_{\text{in}}\in\mathbb{R}^{d\times d_{\text{in}}}.
$$

## 3) BoseFormer Encoder (Transformer Over Occupied Particles)

Use $L$ pre-norm encoder blocks over particle tokens. For block $\ell$:

$$
\tilde{\mathbf{h}}^{(\ell)} = \operatorname{LN}\!\left(\mathbf{h}^{(\ell)}\right).
$$

For each head $a=1,\dots,H$, $d_h=d/H$:

$$
\mathbf{q}^{(\ell,a)}_p = W_Q^{(\ell,a)} \tilde{\mathbf{h}}^{(\ell)}_p,\quad
\mathbf{k}^{(\ell,a)}_p = W_K^{(\ell,a)} \tilde{\mathbf{h}}^{(\ell)}_p,\quad
\mathbf{v}^{(\ell,a)}_p = W_V^{(\ell,a)} \tilde{\mathbf{h}}^{(\ell)}_p.
$$

Scaled dot-product attention:

$$
s^{(\ell,a)}_{pq} = \frac{\mathbf{q}^{(\ell,a)}_p \cdot \mathbf{k}^{(\ell,a)}_q}{\sqrt{d_h}},
$$
$$
\alpha^{(\ell,a)}_{pq} = \operatorname{softmax}_{q}\!\big(s^{(\ell,a)}_{pq}\big),
$$
$$
\mathbf{m}^{(\ell,a)}_p = \sum_{q=1}^{N_f}\alpha^{(\ell,a)}_{pq}\mathbf{v}^{(\ell,a)}_q.
$$

Concatenate heads and project:

$$
\mathbf{m}^{(\ell)}_p = W_O^{(\ell)}\operatorname{concat}_a\mathbf{m}^{(\ell,a)}_p.
$$

Residual + MLP:

$$
\mathbf{y}^{(\ell)}_p = \mathbf{h}^{(\ell)}_p + \mathbf{m}^{(\ell)}_p,
$$
$$
\mathbf{h}^{(\ell+1)}_p = \mathbf{y}^{(\ell)}_p + \operatorname{MLP}^{(\ell)}\!\left(\operatorname{LN}\!\left(\mathbf{y}^{(\ell)}_p\right)\right),
$$

with

$$
\operatorname{MLP}(\mathbf{z}) = W_2 \,\mathrm{GELU}(W_1\mathbf{z}+\mathbf{b}_1)+\mathbf{b}_2.
$$

Notes:
- This follows the PsiFormer attention pattern (self-attention over particle-coordinate tokens), adapted to lattice/PBC inputs.
- We do not need the molecular envelope constraint $\exp(-\sigma\lvert r-R_I\rvert)$ from continuum chemistry, since the lattice domain is finite/PBC.

## 4) Orbital Head And Bosonic Product Aggregation

### Recommended BoseFormer scalar-orbital head

Predict one complex log-orbital per particle from the final token state:

$$
\ell_p = \ell^{(\Re)}_p + i\,\ell^{(\Im)}_p,\qquad
\begin{bmatrix}\ell^{(\Re)}_p\\\ell^{(\Im)}_p\end{bmatrix}
= W_{\text{orb}}\mathbf{h}^{(L)}_p + \mathbf{b}_{\text{orb}}.
$$

Define

$$
\Psi_{\text{BoseFormer}}(n)=\prod_{p=1}^{N_f}\exp(\ell_p),
\quad\Rightarrow\quad
\log\Psi_{\text{BoseFormer}}(n)=\sum_{p=1}^{N_f}\ell_p.
$$

This directly implements "product over orbitals" while remaining numerically stable in log-space.

### Optional richer head

If more expressive capacity is needed, let each particle output $K$ complex features and use a log-cosh reduction:

$$
\ell_p = \sum_{k=1}^{K}\log\cosh\!\big(u_{pk}+i v_{pk}\big),
$$
then sum over $p$. This mirrors the current `ComplexLogCoshOutputHead` style.

## 5) Final BoseFormer x Slater Ansatz

Keep existing Slater branch:

$$
\Psi_{\text{Slater}}(n)=\det\!\big(M_{I(n),:}\big),
\qquad
\log\Psi_{\text{Slater}}(n)=\log\det\!\big(M_{I(n),:}\big).
$$

Compose multiplicatively:

$$
\Psi_{\text{total}}(n)=\Psi_{\text{Slater}}(n)\,\Psi_{\text{BoseFormer}}(n),
$$
$$
\log\Psi_{\text{total}}(n)=\log\Psi_{\text{Slater}}(n)+\log\Psi_{\text{BoseFormer}}(n).
$$

This reuses the same high-level Slater-factor composition pattern as `LogSlaterSpatialViT`, but with a fundamentally different encoder input representation (particle-coordinate tokens instead of site-occupation tokens).

## 6) Proposed Module/API Layout

1. `aidan_custom/models/boseformer.py`
   - `OccupiedPeriodicFeatureEmbedding`
   - `BoseFormerEncoderBlock`
   - `BoseFormerEncoder`
   - `LogBoseFormerProduct`
2. `aidan_custom/models/slater_boseformer.py`
   - `LogSlaterBoseFormer`
3. `aidan_custom/models/__init__.py`
   - Export new classes.
4. Optional utility in `aidan_custom/models/pair_data.py` or new helper:
   - `make_supercell_reciprocal_vectors_from_graph(graph)` returning $(\mathbf{G}_1,\mathbf{G}_2)$.

Suggested constructor signature for combined model:

```python
LogSlaterBoseFormer(
    hilbert=hi,
    positions=positions_hashable,          # (n_sites, 2)
    G_vectors=G_vectors_hashable,          # (2, 2): [G1, G2]
    num_layers=...,
    d_model=...,
    n_heads=...,
    mlp_hidden_factor=...,
    slater_initial_m_orbitals=...,
    param_dtype=jnp.float64,
)
```

## 7) Performance Plan

1. Precompute periodic site features once:
   $$
   F_i = [\cos(\mathbf{r}_i\!\cdot\!\mathbf{G}_1),\sin(\mathbf{r}_i\!\cdot\!\mathbf{G}_1),\cos(\mathbf{r}_i\!\cdot\!\mathbf{G}_2),\sin(\mathbf{r}_i\!\cdot\!\mathbf{G}_2)]
   $$
   then gather $F_{I(n)}$ per sample. This avoids trig in the hot path.
2. Keep tensors as `(batch, N_f, d_model)` and use batched einsum/matmul.
3. Avoid Python loops over particles in `__call__`; rely on JAX primitives.
4. Keep log-space accumulation for product to avoid overflow/underflow.

Attention complexity is $O(B\,L\,H\,N_f^2\,d_h)$. For current $N_f$ ranges in your jobs, this should be tractable.

## 8) Validation Plan

1. Shape/consistency tests:
   - Output shape `(batch,)`, complex dtype.
   - Deterministic output for repeated calls in eval mode.
2. Physics sanity:
   - For `LogSlaterBoseFormer`, setting BoseFormer head to zero recovers pure Slater.
3. Numerical checks:
   - Finite outputs for random batches.
   - Stable gradients (`jax.grad`) on small systems.
4. Training smoke test:
   - Small `FullSumState` job on $L_x=L_y=2$ converges and logs without NaNs.

## 9) Chosen Defaults For First Implementation

1. **BoseFormer head**: direct complex log-orbital sum (no log-cosh head).
2. **Input harmonics**: no higher harmonics; use only the base 4 periodic features.
3. **Attention variant**: pure QKV self-attention (no extra distance bias/envelope term initially).
4. **Nonlinearity**: GELU in MLP blocks (not tanh).

## 10) Input Choice: Raw $(x,y)$ vs Periodic $\sin/\cos$

Recommendation for PBC runs: keep periodic features as the primary input.

- Raw $(x,y)$ coordinates have a boundary discontinuity under PBC (points adjacent through the boundary can look far apart in raw coordinates).
- $\sin(\mathbf{r}\!\cdot\!\mathbf{G}_\mu),\cos(\mathbf{r}\!\cdot\!\mathbf{G}_\mu)$ are explicitly periodic and identify positions modulo lattice translations, which matches the torus geometry.
- For non-orthogonal lattices (e.g. honeycomb embedding), reciprocal-vector projections are more geometry-robust than raw Cartesian axes.

If you want an ablation later, the clean comparison is:

1. periodic-only features (default first run),
2. raw-only features,
3. concatenated periodic + raw features.

## References

- PsiFormer paper: https://arxiv.org/pdf/2211.13672
- DeepMind FermiNet repository (PsiFormer implementation lives there): https://github.com/google-deepmind/ferminet
