# PsiFormer Architecture Plan

## Overview

PsiFormer is a neural quantum state ansatz for lattice fermions. Like BoseFormer x Slater, it uses a Transformer encoder over **occupied-particle coordinate tokens** with periodified Cartesian inputs on a PBC lattice. The key difference is in the output:

- **BoseFormer x Slater**: Transformer outputs one complex scalar per particle → bosonic product → multiplied by a separate, fixed-orbital Slater determinant.
- **PsiFormer**: Transformer outputs $N_f$ complex numbers per particle → these form the **rows of a neural Slater matrix** → the determinant encodes fermionic antisymmetry directly. No separate Slater factor is needed.

The full ansatz is

$$
\log \Psi_\text{total}(n) = \log \Psi_\text{PsiFormer}(n) + \underbrace{\log \Psi_\text{Jastrow}(n)}_{\text{optional}},
$$

where

$$
\Psi_\text{PsiFormer}(n) = \sum_{k=1}^{K} c_k \, \det\!\big(\Phi^{(k)}(n)\big).
$$

$\Phi^{(k)}(n)$ is a configuration-dependent $N_f \times N_f$ complex matrix whose rows are produced by the Transformer, $c_k \in \mathbb{C}$ are learnable mixing coefficients, and $K$ is the number of determinants. Setting $K=1$ and $c_1=1$ recovers the single-determinant special case.

---

## 0) Notation

- Configuration: $n \in \{0,1\}^{N_s}$, fixed particle number $N_f$.
- Occupied indices (sorted): $I(n) = (i_1,\dots,i_{N_f})$, where $n_{i_p}=1$.
- Site positions: $\mathbf{r}_i \in \mathbb{R}^2$, $i=1,\dots,N_s$.
- Primitive reciprocal vectors $(\mathbf{b}_1, \mathbf{b}_2)$, supercell reciprocal vectors $\mathbf{G}_\mu = \mathbf{b}_\mu / L_\mu$.
- Transformer model dimension: $d$.
- Number of determinants: $K$ (single determinant: $K=1$).

---

## 1) Occupied-Particle Tokens

Same as BoseFormer. Extract the $N_f$ occupied coordinates

$$
\mathbf{x}_p = \mathbf{r}_{i_p}, \quad p = 1, \dots, N_f,
$$

using a fixed-size `jnp.nonzero` gather (JIT-friendly, no variable-length control flow).

---

## 2) Input Embedding (Two Options)

Two embedding strategies are supported, selected by the `embedding_type` constructor argument (`"periodic"` by default).

### 2a) Periodic Feature Embedding (`embedding_type="periodic"`)

Same as BoseFormer. For each occupied particle $p$, form a 4-dimensional periodic feature vector from the real-space coordinates:

$$
\mathbf{u}_p =
\begin{bmatrix}
\cos(\mathbf{x}_p \cdot \mathbf{G}_1) \\
\sin(\mathbf{x}_p \cdot \mathbf{G}_1) \\
\cos(\mathbf{x}_p \cdot \mathbf{G}_2) \\
\sin(\mathbf{x}_p \cdot \mathbf{G}_2)
\end{bmatrix}
\in \mathbb{R}^4,
$$

then project linearly to model dimension:

$$
\mathbf{h}^{(0)}_p = W_\text{in} \mathbf{u}_p + \mathbf{b}_\text{in}, \qquad W_\text{in} \in \mathbb{R}^{d \times 4}.
$$

**Precomputation:** All four trig features for every site are computed once at `setup()` time, avoiding trig in the hot path. Per-call, only the gather over $I(n)$ is needed.

Requires `positions` (shape $(N_s, 2)$) and `g_vectors` (shape $(2,2)$) at construction time.

### 2b) Learned Site-Index Embedding (`embedding_type="site_index"`)

Each site $i \in \{0, \dots, N_s - 1\}$ is assigned a learnable embedding vector. A trainable table $E \in \mathbb{R}^{N_s \times d}$ is allocated, and the initial tokens for the occupied particles are simply:

$$
\mathbf{h}^{(0)}_p = E[i_p], \qquad p = 1, \dots, N_f,
$$

where $i_p$ is the 0-based index of the $p$-th occupied site (sorted ascending).

The table $E$ is initialized with Xavier uniform and is jointly optimized with the rest of the network. This embedding carries **no explicit spatial prior** — the network must learn all site-to-site relationships from attention alone.

**Pros:** no geometry input needed; potentially more flexible in systems where real-space coordinates are not the natural prior.
**Cons:** $N_s \times d$ extra parameters; loses built-in translation equivariance; may need more training to learn spatial structure.

Parameter count: $N_s \times d$ reals (versus $4d$ reals for the periodic projection weights, plus $d$ bias terms).

---

## 3) BoseFormer Encoder (Shared With BoseFormer)

Run $L$ pre-norm Transformer blocks over the $N_f$ particle tokens. For block $\ell$, pre-normalize, then apply QKV self-attention followed by a residual add, then a pre-norm MLP block:

$$
\tilde{\mathbf{h}}^{(\ell)} = \operatorname{LN}\!\left(\mathbf{h}^{(\ell)}\right),
$$

$$
\mathbf{m}^{(\ell)}_p = W_O^{(\ell)} \operatorname{concat}_a \left[ \sum_{q=1}^{N_f} \operatorname{softmax}_q\!\left(\frac{\mathbf{q}^{(\ell,a)}_p \cdot \mathbf{k}^{(\ell,a)}_q}{\sqrt{d_h}}\right) \mathbf{v}^{(\ell,a)}_q \right],
$$

$$
\mathbf{y}^{(\ell)}_p = \mathbf{h}^{(\ell)}_p + \mathbf{m}^{(\ell)}_p,
$$

$$
\mathbf{h}^{(\ell+1)}_p = \mathbf{y}^{(\ell)}_p + \operatorname{MLP}^{(\ell)}\!\left(\operatorname{LN}\!\left(\mathbf{y}^{(\ell)}_p\right)\right),
$$

with $\operatorname{MLP}(\mathbf{z}) = W_2\,\operatorname{GELU}(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2$. This uses the existing `BoseFormerEncoder` module from `boseformer.py` unchanged.

**Permutation equivariance:** The Transformer is permutation-equivariant in its token set. If the occupied indices are permuted $I(n) \to \sigma(I(n))$, the output tokens permute in the same way:

$$
\mathbf{h}^{(L)}_{\sigma(p)} = f_\theta\!\left(\mathbf{x}_{\sigma(p)}\ \big|\ \{\mathbf{x}_q\}\right).
$$

This is the key property that guarantees exact antisymmetry below.

---

## 4) Neural Orbital Head

**Goal:** Map each token $\mathbf{h}^{(L)}_p \in \mathbb{R}^d$ to $K \times N_f$ complex numbers, forming the $p$-th row of each of the $K$ neural orbital matrices.

Apply a final layer norm, then two independent linear projections (real and imaginary parts):

$$
\mathbf{z}_p = \operatorname{LN}_\text{out}\!\left(\mathbf{h}^{(L)}_p\right),
$$

$$
\phi^{(\Re)}_p = W_\Re \mathbf{z}_p \in \mathbb{R}^{K N_f}, \qquad
\phi^{(\Im)}_p = W_\Im \mathbf{z}_p \in \mathbb{R}^{K N_f},
$$

$$
\boldsymbol{\phi}_p = \phi^{(\Re)}_p + i\,\phi^{(\Im)}_p \in \mathbb{C}^{K N_f}.
$$

Reshape to separate determinant and orbital indices:

$$
\boldsymbol{\phi}_p \to \Phi_{p,:,:} \in \mathbb{C}^{K \times N_f},
$$

so that $\Phi^{(k)}_{p,j} = \phi_{p}^{(k,j)}$ is the amplitude of orbital $j$ for particle $p$ in determinant $k$.

**Parameter count of the head:** $2 \times K N_f \times d$ (real), versus $2 \times d$ for the BoseFormer product head. This grows with $K$ and $N_f$ but is typically small compared to the encoder.

---

## 5) Multi-Determinant Aggregation

Stack the rows over particles to form $K$ square matrices:

$$
\Phi^{(k)} \in \mathbb{C}^{N_f \times N_f}, \qquad \Phi^{(k)}_{p,j} = \text{orbital } j \text{ evaluated at particle } p, \text{ determinant } k.
$$

Compute the complex log-determinant for each:

$$
\ell_k = \log \det\!\big(\Phi^{(k)}\big) = \log\lvert\det\Phi^{(k)}\rvert + i \arg\!\big(\det\Phi^{(k)}\big),
$$

implemented numerically as `slogdet` + complex recombination (same as the existing `_logdet_cmplx`).

Sum with learnable mixing coefficients $c_k \in \mathbb{C}$:

$$
\Psi_\text{PsiFormer}(n) = \sum_{k=1}^K c_k \exp(\ell_k).
$$

Taking the complex log:

$$
\log \Psi_\text{PsiFormer}(n) = \operatorname{logaddexp}_{\mathbb{C}}\!\left(\log c_k + \ell_k\right)_{k=1}^K,
$$

i.e., implemented as a complex logsumexp over the $K$ terms. For $K=1$ (single determinant), this reduces to $\log c_1 + \ell_1$, and if $c_1$ is fixed to 1 the result is simply $\ell_1$.

**Initialization:** Initialize $W_\Re$ with small random values (Xavier), $W_\Im$ to zero, and $c_k = 1/K$ (or $c_1=1$ for $K=1$). This makes PsiFormer start close to a pure determinant.

---

## 6) Fermionic Antisymmetry (Why No Separate Slater Factor Is Needed)

For any transposition $\tau$ of two particle labels $p \leftrightarrow p'$:

1. **Encoder equivariance:** the output tokens permute, $\mathbf{h}^{(L)}_p \leftrightarrow \mathbf{h}^{(L)}_{p'}$.
2. **Rows of $\Phi^{(k)}$ swap:** rows $p$ and $p'$ are exchanged.
3. **Determinant is antisymmetric:** $\det(\text{row-swapped matrix}) = -\det(\text{original})$.
4. **Each term $c_k \det(\Phi^{(k)})$ flips sign**, so their sum does too.

Therefore $\Psi_\text{PsiFormer}(n)$ is **exactly antisymmetric** under any particle permutation, by construction, without any external Slater factor.

---

## 7) Optional Jastrow Factor

To add variational flexibility cheaply, add NetKet's built-in two-body Jastrow:

$$
\log \Psi_\text{total}(n) = \log \Psi_\text{PsiFormer}(n) + \log \Psi_\text{Jastrow}(n),
$$

where $\Psi_\text{Jastrow}$ is `nk.models.Jastrow` (already used in `LogSlaterJastrow`). Since the Jastrow factor is symmetric under particle exchange, it does not spoil antisymmetry. This is toggled with a boolean constructor argument `use_jastrow: bool`.

---

## 8) Full Ansatz Summary

$$
\boxed{
\log \Psi_\text{total}(n) = \log\!\left(\sum_{k=1}^K c_k \det\!\big(\Phi^{(k)}(n)\big)\right) + \mathbf{1}[\text{use\_jastrow}]\cdot\log \Psi_\text{Jastrow}(n)
}
$$

where $\Phi^{(k)}(n)$ is computed by (two embedding options):

$$
n \;\xrightarrow{\text{gather}}\; \{i_p\} \;\xrightarrow{\begin{cases}\text{periodic embed} \\ \text{site-index embed}\end{cases}}\; \mathbf{h}^{(0)} \;\xrightarrow{\text{BoseFormer encoder}}\; \mathbf{h}^{(L)} \;\xrightarrow{\text{LN + linear head}}\; \Phi^{(k)}.
$$

| `embedding_type` | Input to encoder | Spatial prior | Extra params |
|---|---|---|---|
| `"periodic"` | $\cos/\sin$ of $\mathbf{r}_{i_p} \cdot \mathbf{G}_\mu$, projected | Yes (PBC-aware) | $4d + d$ ($W_\text{in}$, $\mathbf{b}_\text{in}$) |
| `"site_index"` | Row $i_p$ of learned table $E \in \mathbb{R}^{N_s \times d}$ | None | $N_s \cdot d$ |

---

## 9) Comparison With Prior Architectures

| Property | Slater | BoseFormer × Slater | **PsiFormer** |
|---|---|---|---|
| Antisymmetry source | Fixed orbital matrix $M$ | Fixed Slater × bosonic product | Neural orbital matrix $\Phi$ |
| Transformer output | — | 1 complex scalar per particle | $K \cdot N_f$ complex values per particle |
| Aggregation | `det` | product × `det` | `det` (or sum of `det`s) |
| Separate Slater factor | ✓ (is the whole model) | ✓ | ✗ |
| Orbital flexibility | Low (static $M$) | Low static + flexible corr. | High (fully dynamic $\Phi$) |
| Parameters (head) | $2 N_s N_f$ | $2d$ (BF) + $2 N_s N_f$ (Slater) | $2 K N_f d$ + $2K$ (coeffs) |
| Complexity (hot path) | $O(N_f^3)$ | $O(L H N_f^2 d_h + N_f^3)$ | $O(L H N_f^2 d_h + K N_f^3)$ |

---

## 10) Proposed Module / API Layout

```
aidan_custom/models/
  psiformer.py                   ← new file
    NeuralOrbitalHead            # token (batch, N_f, d) → orbital matrix (batch, K, N_f, N_f)
    LogPsiFormer                 # full standalone ansatz (encoder + head + det)
  slater_psiformer.py            ← new file (if combined with Jastrow or other factors)
    LogPsiFormerJastrow          # LogPsiFormer + nk.models.Jastrow
  __init__.py                    ← export new classes
```

### Constructor signature for standalone model

```python
LogPsiFormer(
    hilbert: nk.hilbert.SpinOrbitalFermions,
    positions: Any,          # (n_sites, 2); None allowed when embedding_type="site_index"
    g_vectors: Any,          # (2, 2): [G1, G2]; None allowed when embedding_type="site_index"
    num_layers: int,
    d_model: int,
    n_heads: int,
    n_determinants: int = 1,
    mlp_hidden_factor: int = 4,
    embedding_type: str = "periodic",  # "periodic" or "site_index"
    use_jastrow: bool = False,
    jastrow_param_dtype: DType = jnp.float64,
    param_dtype: DType = jnp.float64,
    kernel_init: Any = nn.initializers.xavier_uniform(),
    orbital_kernel_init: Any = nn.initializers.xavier_uniform(),
    det_coeff_init: Any = nn.initializers.ones,    # c_k initialized to 1
)
```

---

## 11) Numerical Implementation Notes

### Complex log-determinant
Use `_logdet_cmplx` (already in `slater.py`):
```python
sign, logabsdet = jnp.linalg.slogdet(matrix)
return logabsdet.astype(complex) + jnp.log(sign.astype(complex))
```

### Multi-determinant logsumexp
For $K > 1$, avoid forming $\sum_k c_k \det(\Phi^{(k)})$ directly (overflow). Instead compute the complex logsumexp:
```python
log_dets   # shape (batch, K), complex
log_coeffs = jnp.log(c.astype(complex))   # shape (K,)
terms = log_dets + log_coeffs[None, :]    # (batch, K)
# stable complex logsumexp:
m = jnp.real(terms).max(axis=-1, keepdims=True)
log_psi = m[..., 0] + jnp.log(jnp.sum(jnp.exp(terms - m), axis=-1))
```

### Orbital head split-complex convention
Follow the existing `split_complex_params` style: store $W_\Re, W_\Im$ as separate real parameters; combine with `jax.lax.complex` at call time. This keeps params real-valued (better for optimizer stats and checkpointing).

### Initialization intent
- `orbital_kernel_init = xavier_uniform` (not zeros): unlike the BoseFormer product head, the head output goes into a determinant, so an all-zero initialization produces a singular matrix. Xavier or similar gives a well-conditioned random Slater matrix at init.
- `det_coeff_init = ones`: all determinants contribute equally at the start.

---

## 12) Validation Plan

1. **Antisymmetry unit test:** swap two occupied sites in a configuration; verify `log_psi` changes by exactly $i\pi$ (sign flip).
2. **Shape test:** output shape `(batch,)`, complex dtype.
3. **$K=1$ limit:** confirm single-determinant PsiFormer gives a valid antisymmetric wavefunction.
4. **Finite gradients:** `jax.grad` on a small system produces no NaNs.
5. **Training smoke test:** small `FullSumState` run on $L_x = L_y = 2$ converges and logs without NaNs.
6. **Comparison baseline:** for the same system, compare variational energy against BoseFormer × Slater.

---

## 13) Defaults for First Implementation

1. $K = 1$ (single determinant) — upgrade to multi-det once single works.
2. No Jastrow (toggle off by default).
3. Linear-only orbital head (no log-cosh or nonlinearity before the projection).
4. GELU in MLP blocks, same as BoseFormer.
5. `split_complex_params = True` (real params + `jax.lax.complex`).
6. `orbital_kernel_init = xavier_uniform` (random, not zeros).

---

## References

- PsiFormer (Scherbela et al., 2022): https://arxiv.org/pdf/2211.13672
- FermiNet (Pfau et al., 2020): https://arxiv.org/abs/1909.02487
- BoseFormer × Slater plan: `aidan_custom/notes/BoseFormer_x_Slater.md`
