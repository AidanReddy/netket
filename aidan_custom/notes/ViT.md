## Spatial ViT Architecture (Exactly As Implemented)

This describes the **ViT factor** in `LogSlaterSpatialViT`, i.e. $\log\Psi_{\mathrm{ViT}}$, exactly as implemented in `aidan_custom/models/`.

The full ansatz used in `LogSlaterSpatialViT` is

$$
\log\Psi(n)=\log\Psi_{\mathrm{Slater}}(n)+\log\Psi_{\mathrm{ViT}}(n).
$$

### 1) Input And Site Embedding

For spinless occupations $(n\in\{0,1\}^{N_s})$, each site is a token. Before embedding we remap occupancy to Ising-style values:

$$
s_i = \begin{cases}
+1 & n_i > 0 \\
-1 & n_i \le 0
\end{cases}
$$

A shared dense map then embeds each scalar $s_i$ into $(d\equiv d_{\text{model}})$:

$$
\mathbf{x}^{(0)}_i = W_{\mathrm{emb}}\, s_i + \mathbf{b}_{\mathrm{emb}},
\qquad W_{\mathrm{emb}}\in\mathbb{R}^{1\times d},\; \mathbf{b}_{\mathrm{emb}}\in\mathbb{R}^{d}.
$$

### 2) Translation-Equivariant Pair Classes (Minimum-Image)

Using site positions $(\mathbf{r}_i)$, lattice basis vectors $((\mathbf{a}_1,\mathbf{a}_2))$, extents $((L_x,L_y))$, and periodic flags, translations are

$$
\mathbf{t}_{m_1,m_2}=(m_1L_x)\mathbf{a}_1 + (m_2L_y)\mathbf{a}_2,
$$

with $(m_\mu\in\{-1,0,1\})$ if periodic in direction $(\mu)$, else $(m_\mu=0)$.

For each pair $(i,j)$, the displacement used by attention is the minimum-image vector

$$
\Delta\mathbf{r}_{ij} = \arg\min_{\mathbf{t}}\left\| (\mathbf{r}_i-\mathbf{r}_j)+\mathbf{t} \right\|_2.
$$

At Wigner-Seitz boundaries this minimizer can be non-unique. In implementation, ties are broken deterministically by choosing the first translation in the ordered scan $(m_1,m_2)\in(-1,0,1)\times(-1,0,1)$, which gives a single canonical representative per periodic-equivalence class.

These vectors are rounded (default 12 decimals), uniqued into classes $c\in\{1,\dots,K\}$, and mapped by

$$
C_{ij}\in\{1,\dots,K\},\qquad d_c=\|\Delta\mathbf{r}_c\|_2.
$$

So each pair $(i,j)$ references a class index $C_{ij}$, and each class has a fixed distance $d_c$.

For Bravais lattices this gives $K=L_xL_y$ classes. For lattices with a basis, $K$ can be larger because distinct sublattice-offset differences can yield distinct minimum-image vectors (and therefore distinct classes).

### 3) One Spatial Attention Layer (Factored Kernel)

For an input $\mathbf{x}\in\mathbb{R}^{B\times N_s\times d}$:

1. **Value projection only** (no query/key projections):

$$
\mathbf{v}_i = W_V\mathbf{x}_i + \mathbf{b}_V,
\qquad W_V\in\mathbb{R}^{d\times d},\;\mathbf{b}_V\in\mathbb{R}^{d}.
$$

2. Split into $H$ heads, $d_h=d/H$: $\mathbf{v}_i\to\mathbf{v}_i^{(a)}\in\mathbb{R}^{d_h}$.

3. Learned per-head/class amplitudes $\alpha_{a,c}$ and per-head length scales $\xi_a$:

$$
\xi_a = \operatorname{softplus}(\rho_a)+\epsilon_\xi,
$$

where $\rho_a$ is `raw_xi`, $\epsilon_\xi=\texttt{vit\_xi\_epsilon}$, and default initialization sets $\xi_a\approx3.0$ for all heads.

4. Unnormalized exponential envelope:

$$
\widehat{E}_{a,c}=\exp\!\left(-\frac{d_c}{\xi_a}\right).
$$

5. Pairwise normalization (for each head $a$ and destination site $i$):

$$
Z_{a,i}=\sum_{j'=1}^{N_s}\widehat{E}_{a,\,C_{ij'}},
\qquad
E_{a,ij}=\frac{\widehat{E}_{a,\,C_{ij}}}{Z_{a,i}}.
$$

6. Pair kernel with learned amplitude:

$$
A_{a,ij}=\alpha_{a,\,C_{ij}}\,E_{a,ij}.
$$

7. Headwise aggregation (no query-key softmax):

$$
\tilde{\mathbf{v}}^{(a)}_i = \sum_{j=1}^{N_s} A_{a,ij}\,\mathbf{v}^{(a)}_j.
$$

8. Concatenate heads and apply output projection:

$$
\tilde{\mathbf{v}}_i = \operatorname{concat}_a\tilde{\mathbf{v}}^{(a)}_i\in\mathbb{R}^d,
\qquad
\mathbf{y}_i = W_O\tilde{\mathbf{v}}_i + \mathbf{b}_O,
$$

with $W_O\in\mathbb{R}^{d\times d}$, $\mathbf{b}_O\in\mathbb{R}^d$.

### 4) Encoder Block (Pre-LN, Residual, FFN)

Each block applies

$$
\mathbf{y} = \mathbf{x} + \operatorname{Attn}(\operatorname{LN}_1(\mathbf{x})),
$$

$$
\mathbf{z} = \mathbf{y} + \operatorname{FF}(\operatorname{LN}_2(\mathbf{y})),
$$

with feature-wise FFN

$$
\operatorname{FF}(\mathbf{u}) = W_2\,\operatorname{GELU}(W_1\mathbf{u}+\mathbf{b}_1)+\mathbf{b}_2,
$$

$$
W_1\in\mathbb{R}^{d\times(rd)},\quad W_2\in\mathbb{R}^{(rd)\times d},\quad r=\texttt{vit\_mlp\_hidden\_factor}.
$$

Stacking $L=\texttt{vit\_num\_layers}$ such blocks gives $\mathbf{x}^{(L)}$.

### 5) Complex Output Head For $\log\Psi_{\mathrm{ViT}}$

Pool over sites and normalize:

$$
\mathbf{z}=\operatorname{LN}_{\mathrm{out}}\!\left(\sum_{i=1}^{N_s} \mathbf{x}^{(L)}_i\right)\in\mathbb{R}^d.
$$

Real and imaginary branches:

$$
\mathbf{r}=\operatorname{LN}_{\Re}(W_{\Re}\mathbf{z}+\mathbf{b}_{\Re})\in\mathbb{R}^{d_o},
\qquad
\mathbf{u}=\operatorname{LN}_{\Im}(W_{\Im}\mathbf{z}+\mathbf{b}_{\Im})\in\mathbb{R}^{d_o},
$$

where $d_o=\texttt{vit\_output\_hidden\_dim}$ (or $d$ if `None`).

Finally,

$$
\log\Psi_{\mathrm{ViT}}(n)=\sum_{k=1}^{d_o}\log\cosh\!\big(r_k+i\,u_k\big).
$$

### 6) Dtype/Complexity Convention

- All ViT internal parameters and activations are real (`vit_param_dtype`, currently float64 by default).
- Complex structure appears only at the final combination $\mathbf{r}+i\mathbf{u}$ in the output head.
- There are no query/key projections and no query-key softmax in this implementation; attention weights are $A_{a,ij}=\alpha_{a,C_{ij}}\,\widehat{E}_{a,C_{ij}}/\sum_{j'}\widehat{E}_{a,C_{ij'}}$ with $\widehat{E}_{a,c}=\exp(-d_c/\xi_a)$.
