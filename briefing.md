---
tags: projects, agents
---

# Tensor Network FutureLens: Coding-Agent Briefing

**Project owners:** Aniket Deshpande and Dmitry Manning-Coe  
**Working title:** Tensor Network FutureLens  
**Status:** early-stage research prototype  
**Audience for this document:** a coding agent such as Claude Code, Codex, or another autonomous repo-building agent.

This document describes the project, the theory motivation, the experiments, the implementation plan, the expected artifacts, and the coding standards. It is intentionally detailed. The point is not just to produce a working model. The point is to execute a **physicist's workflow**: define the theoretical object, derive the observables, measure the assumptions, build the minimal model, run controlled experiments, and interpret every result in relation to both transformer residual-stream theory and many-body tensor-network theory.

Use Markdown math delimiters `$...$` and `$$...$$` throughout generated notes and reports. Do **not** use `\(...\)` or `\[...\]` in Markdown outputs.

---

## 0. Executive summary

FutureLens studies whether a **single hidden state** in a transformer contains enough information to anticipate future tokens. It trains/constructs probes that map one hidden/residual state at token position $t$ and layer $\ell$ to predictions about future residual states or future token distributions.

Tensor Network FutureLens generalizes this by replacing the single hidden state with a **local residual-stream trajectory**:

$$
\mathcal R_t^{\ell;m,n}
=
\left(
r_{t-m+1}^{\ell},
\ldots,
r_t^\ell,
r_{t+1}^{\ell},
\ldots,
r_{t+n}^{\ell}
\right).
$$

The first $m$ residual-stream sites are observed. The next $n$ future residual-stream sites are to be completed/predicted.

The physics hypothesis is:

> If transformer residual-stream features along token position have finite correlation length, then a one-dimensional tensor network, especially an MPS, should efficiently model the local residual trajectory. The MPS transfer matrix has dimension $D^2$ when the bond dimension is $D$, so a bond-$D$ MPS can represent up to $O(D^2)$ exponential correlation modes in two-point functions.

The empirical program is:

1. Measure residual-stream two-point correlations.
2. Test whether they decay exponentially or as a small sum of exponentials.
3. Reproduce single-state FutureLens-style baselines.
4. Train multi-site linear/MLP/attention baselines.
5. Train MPS/Tensor-Train probes with varying physical dimension $p$ and bond dimension $D$.
6. Ask whether MPS improvements track the transfer-matrix theory.

A good result is **not** merely "MPS beats single-state FutureLens." That could just mean "more context helps." A good result is:

$$
\text{MPS beats matched-parameter multi-site baselines at horizons where the measured residual correlations are well fit by few exponential modes.}
$$

---

## 1. Papers and resources to read / cite

The coding agent may websearch as needed. Use the following as anchor references.

### FutureLens

- **Future Lens: Anticipating Subsequent Tokens from a Single Hidden State**  
  Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, David Bau. CoNLL 2023.  
  Paper: <https://aclanthology.org/2023.conll-1.37/>  
  arXiv: <https://arxiv.org/abs/2311.04897>  
  Project page: <https://future.baulab.info/>  
  Repo: <https://github.com/KoyenaPal/future-lens>

Important concept: FutureLens asks whether one hidden state at one token/layer can anticipate tokens several positions ahead. It evaluates linear approximation and causal intervention methods.

### Tensor networks for ML

- **Supervised Learning with Quantum-Inspired Tensor Networks**  
  E. Miles Stoudenmire and David J. Schwab, NeurIPS 2016.  
  arXiv: <https://arxiv.org/abs/1605.05775>

Important concept: a local feature map sends each input component into a physical site vector, and an MPS/Tensor Train parameterizes a high-order weight tensor efficiently.

### MPS theory

- **Matrix Product State Representations**  
  D. Pérez-García, F. Verstraete, M. M. Wolf, J. I. Cirac.  
  arXiv: <https://arxiv.org/abs/quant-ph/0608197>

Important concepts: MPS representations, canonical forms, gauge freedom, efficient classical representation.

### Exponential decay and area-law intuition

- **Exponential Decay of Correlations Implies Area Law**  
  F. G. S. L. Brandão and M. Horodecki.  
  arXiv: <https://arxiv.org/abs/1206.2947>

Important concept: in one-dimensional many-body systems, finite correlation length/exponential correlation decay is deeply connected to efficient MPS descriptions.

### Transformer and activation extraction tooling

- **TransformerLens**  
  Docs: <https://transformerlensorg.github.io/TransformerLens/>  
  GitHub: <https://github.com/TransformerLensOrg/TransformerLens>

Important concept: use TransformerLens to load open-source transformers and cache internal residual-stream activations.

### Tensor-network tooling

- **TensorNetwork: A Library for Physics and Machine Learning**  
  arXiv: <https://arxiv.org/abs/1905.01330>  
  GitHub: <https://github.com/google/TensorNetwork>

Use this as background. For implementation, PyTorch/einsum may be simpler than depending on TensorNetwork.

---

## 2. Conceptual framing

### 2.1 The transformer object

Let $F_\Theta$ be a frozen autoregressive transformer. Let $\mathcal V$ be the vocabulary and let $x_{1:T}$ be a token sequence:

$$
x_{1:T}=(x_1,\ldots,x_T), \qquad x_i\in \mathcal V.
$$

Let $d_{\mathrm{model}}$ be the model width and $L$ the number of layers. The residual stream vector at token position $i$ and layer $\ell$ is

$$
r_i^\ell \in \mathbb R^{d_{\mathrm{model}}},
\qquad
i=1,\ldots,T,
\qquad
\ell=0,\ldots,L.
$$

The final next-token distribution is

$$
p_\Theta(\cdot\mid x_{\le i})
=
\operatorname{softmax}
\left(
W_U \operatorname{LN}(r_i^L)+b_U
\right).
$$

Important off-by-one convention:

$$
r_i^L \quad \text{predicts} \quad x_{i+1}.
$$

Therefore a predicted final-layer residual $\widehat r_{t+s}^L$ corresponds to a predicted distribution over token $x_{t+s+1}$.

### 2.2 FutureLens baseline

A single-state linear FutureLens-style predictor is

$$
\widehat r_{t+s}^{L}
=
A_{\ell,s} r_t^\ell+b_{\ell,s}.
$$

Then

$$
q_{\ell,s}(\cdot\mid r_t^\ell)
=
\operatorname{softmax}
\left(
W_U \operatorname{LN}(A_{\ell,s}r_t^\ell+b_{\ell,s})+b_U
\right).
$$

Possible losses:

Residual regression:

$$
\mathcal L_{\mathrm{res}}
=
\mathbb E_{x,t}
\left[
\left\|
A_{\ell,s}r_t^\ell+b_{\ell,s}-r_{t+s}^L
\right\|_2^2
\right].
$$

Teacher KL:

$$
\mathcal L_{\mathrm{KL}}
=
\mathbb E_{x,t}
\left[
\operatorname{KL}
\left(
p_\Theta(\cdot\mid x_{\le t+s})
\middle\|
q_{\ell,s}(\cdot\mid r_t^\ell)
\right)
\right].
$$

Realized-token cross entropy:

$$
\mathcal L_{\mathrm{CE}}
=
\mathbb E_{x,t}
\left[
-\log q_{\ell,s}(x_{t+s+1}\mid r_t^\ell)
\right].
$$

### 2.3 Tensor Network FutureLens object

Define the local residual-stream trajectory:

$$
\mathcal R_t^{\ell;m,n}
=
\left(
r_{t-m+1}^{\ell},
\ldots,
r_t^\ell,
r_{t+1}^{\ell},
\ldots,
r_{t+n}^{\ell}
\right).
$$

Here:

- $t$ is the anchor position.
- $\ell$ is the source layer.
- $m$ is the number of observed residual-stream sites.
- $n$ is the number of future residual-stream sites to complete.

This notation has several superscripts/subscripts, but the actual tensor-network geometry is **one-dimensional** for fixed $t,\ell,m,n$:

$$
r_{t-m+1}^\ell
-
r_{t-m+2}^\ell
-
\cdots
-
r_t^\ell
-
r_{t+1}^\ell
-
\cdots
-
r_{t+n}^\ell.
$$

This is why the default architecture is an **MPS**, not a PEPS.

A PEPS would only be appropriate if we made the geometry two-dimensional, for example by including both token position and layer:

$$
\{r_{t+j}^{\ell'} : j=-m+1,\ldots,n,\ \ell'\in \Lambda\}.
$$

That would be a position-by-layer grid. The initial project should avoid this and remain one-dimensional.

---

## 3. Physical site map $\phi$

### 3.1 Why we need a physical site map

An MPS site has a physical leg of dimension $p$ and two virtual legs of dimension $D$:

$$
A_j \in \mathbb R^{D\times p\times D}.
$$

Equivalently, for each physical index $a=1,\ldots,p$, the local tensor slice is a matrix:

$$
A_j^a \in \mathbb R^{D\times D}.
$$

A residual vector is

$$
r_j^\ell\in \mathbb R^{d_{\mathrm{model}}}.
$$

The simplest choice is $\phi=\mathrm{id}$, so $p=d_{\mathrm{model}}$. Then the local physical state is

$$
|r_j^\ell\rangle
=
\sum_{a=1}^{d_{\mathrm{model}}}
r_{j,a}^\ell |a\rangle.
$$

Feeding the site into the MPS means contracting the physical leg:

$$
A_j(r_j^\ell)
=
\sum_{a=1}^{d_{\mathrm{model}}}
r_{j,a}^\ell A_j^a.
$$

This is valid.

But the raw residual coordinate basis may be a bad physical basis: large, anisotropic, dense, and affected by superposition. Therefore define a local feature map:

$$
\phi^\ell:\mathbb R^{d_{\mathrm{model}}}\to \mathbb R^p.
$$

Then

$$
v_j^\ell = \phi^\ell(r_j^\ell)\in \mathbb R^p.
$$

The local contraction becomes

$$
A_j(v_j^\ell)
=
\sum_{a=1}^{p}
v_{j,a}^\ell A_j^a.
$$

### 3.2 Feature map options

Implement at least these three:

#### Option 1: identity

$$
\phi(r)=r,\qquad p=d_{\mathrm{model}}.
$$

This is conceptually simplest but expensive.

#### Option 2: PCA/whitening

Estimate empirical mean and covariance:

$$
\mu^\ell = \mathbb E[R^\ell],
$$

$$
\Sigma^\ell
=
\mathbb E[(R^\ell-\mu^\ell)(R^\ell-\mu^\ell)^\top].
$$

Take top-$p$ eigenvectors:

$$
\Sigma^\ell = U\Lambda U^\top,
\qquad
U_p=[u_1,\ldots,u_p].
$$

Define

$$
\phi^\ell(r)
=
(\Lambda_p+\varepsilon I)^{-1/2}
U_p^\top(r-\mu^\ell).
$$

This yields approximately whitened features.

#### Option 3: learned linear map

Define

$$
\phi_\eta(r)
=
W_\phi r + b_\phi,
\qquad
W_\phi\in \mathbb R^{p\times d_{\mathrm{model}}}.
$$

Here $p$ is a hyperparameter, not usually a learned variable. Sweep values such as $p\in\{32,64,128,256\}$.

If $\phi$ is linear, then $\phi$ plus the MPS local tensor is equivalent to an identity physical leg with a constrained/factorized local tensor. Indeed:

$$
A_j(W_\phi r_j)
=
\sum_{a=1}^{p}(W_\phi r_j)_a A_j^a.
$$

Expand:

$$
A_j(W_\phi r_j)
=
\sum_{a=1}^{p}
\sum_{b=1}^{d_{\mathrm{model}}}
(W_\phi)_{ab}r_{j,b}A_j^a.
$$

Swap sums:

$$
A_j(W_\phi r_j)
=
\sum_{b=1}^{d_{\mathrm{model}}}
r_{j,b}
\left(
\sum_{a=1}^{p}(W_\phi)_{ab}A_j^a
\right).
$$

Define

$$
B_j^b
=
\sum_{a=1}^{p}(W_\phi)_{ab}A_j^a.
$$

Then

$$
A_j(W_\phi r_j)
=
\sum_{b=1}^{d_{\mathrm{model}}}
r_{j,b}B_j^b.
$$

Interpretation:

$$
W_\phi \text{ learns the local physical basis, while } D \text{ controls inter-position virtual correlations.}
$$

This is an important ablation: identity versus PCA/whitening versus learned $\phi$.

---

## 4. MPS geometry and transfer matrix

### 4.1 MPS functional over residual sites

For a window of $N=m+n$ sites, after applying $\phi$, we have

$$
v_1,\ldots,v_N,\qquad v_j\in\mathbb R^p.
$$

Define local matrices

$$
A_j(v_j)
=
\sum_{a=1}^{p}v_{j,a}A_j^a,
\qquad
A_j^a\in \mathbb R^{D_{j-1}\times D_j}.
$$

For open boundary conditions:

$$
D_0=D_N=1,
\qquad
D_j\le D.
$$

The scalar MPS functional is

$$
\psi_\theta(v_1,\ldots,v_N)
=
A_1(v_1)A_2(v_2)\cdots A_N(v_N),
$$

where the product is scalar because the boundary dimensions are one. With explicit boundary vectors for a translation-invariant or bulk-shared version:

$$
\psi_\theta(v_1,\ldots,v_N)
=
\ell_{\mathrm{bd}}^\top
A(v_1)A(v_2)\cdots A(v_N)
r_{\mathrm{bd}}.
$$

Expanding:

$$
\psi_\theta(v_1,\ldots,v_N)
=
\sum_{a_1,\ldots,a_N=1}^{p}
W_{a_1,\ldots,a_N}
\prod_{j=1}^N v_{j,a_j},
$$

with MPS coefficient tensor

$$
W_{a_1,\ldots,a_N}
=
\ell_{\mathrm{bd}}^\top
A^{a_1}A^{a_2}\cdots A^{a_N}
r_{\mathrm{bd}}.
$$

### 4.2 Transfer matrix

For a translation-invariant MPS, define

$$
E
=
\sum_{a=1}^{p}A^a\otimes \overline{A^a}.
$$

If $A^a\in\mathbb R^{D\times D}$, then

$$
E\in\mathbb R^{D^2\times D^2}.
$$

For a local observable $O$ with matrix entries $O_{ab}$, define the inserted transfer matrix

$$
E_O
=
\sum_{a,b=1}^{p}
O_{ab} A^b\otimes \overline{A^a}.
$$

The one-point function is

$$
\langle O_i\rangle
=
\frac{
(L|E^{i-1}E_OE^{N-i}|R)
}{
(L|E^N|R)
}.
$$

The two-point function for $i<j$ is

$$
\langle O_iP_j\rangle
=
\frac{
(L|E^{i-1}E_OE^{j-i-1}E_PE^{N-j}|R)
}{
(L|E^N|R)
}.
$$

For an infinite, translation-invariant chain, normalize the leading transfer eigenvalue to $\lambda_1=1$ and write

$$
E
=
\sum_{\mu=1}^{D^2}
\lambda_\mu |r_\mu)(\ell_\mu|.
$$

Then

$$
E^k
=
\sum_{\mu=1}^{D^2}
\lambda_\mu^k |r_\mu)(\ell_\mu|.
$$

Thus

$$
\langle O_0P_\Delta\rangle
=
(\ell_1|E_OE^{\Delta-1}E_P|r_1)
$$

becomes

$$
\langle O_0P_\Delta\rangle
=
\sum_{\mu=1}^{D^2}
\lambda_\mu^{\Delta-1}
(\ell_1|E_O|r_\mu)(\ell_\mu|E_P|r_1).
$$

The $\mu=1$ term equals $\langle O\rangle\langle P\rangle$, so the connected correlation is

$$
\langle O_0P_\Delta\rangle_c
=
\sum_{\mu=2}^{D^2}
c_\mu(O,P)\lambda_\mu^{\Delta-1}.
$$

If $|\lambda_\mu|<1$, write

$$
|\lambda_\mu| = e^{-1/\xi_\mu}.
$$

Then

$$
|\lambda_\mu|^\Delta = e^{-\Delta/\xi_\mu}.
$$

So each nontrivial transfer eigenvalue gives an exponential correlation mode.

Main theoretical prediction:

$$
\text{bond dimension }D
\Rightarrow
\text{transfer dimension }D^2
\Rightarrow
O(D^2)\text{ exponential correlation modes.}
$$

This is the basis for the "quadratic" advantage claim.

---

## 5. Residual-stream correlation diagnostics

### 5.1 What to measure

For fixed layer $\ell$, define centered physical features:

$$
\widetilde V_i^\ell
=
V_i^\ell-\mathbb E[V_i^\ell],
\qquad
V_i^\ell=\phi^\ell(r_i^\ell).
$$

The matrix-valued two-point function is

$$
C^\ell(\Delta)
=
\mathbb E_i
\left[
\widetilde V_i^\ell
(\widetilde V_{i+\Delta}^\ell)^\top
\right]
\in \mathbb R^{p\times p}.
$$

Compute scalar summaries:

$$
c_F^\ell(\Delta)=\|C^\ell(\Delta)\|_F,
$$

$$
c_{\mathrm{op}}^\ell(\Delta)=\|C^\ell(\Delta)\|_{\mathrm{op}},
$$

and, if useful,

$$
c_{\mathrm{tr}}^\ell(\Delta)=\operatorname{Tr}(C^\ell(\Delta)).
$$

If features are not whitened, also compute normalized/whitened correlations:

$$
\widehat C^\ell(\Delta)
=
(C^\ell(0)+\varepsilon I)^{-1/2}
C^\ell(\Delta)
(C^\ell(0)+\varepsilon I)^{-1/2}.
$$

Then summarize by

$$
\|\widehat C^\ell(\Delta)\|_F
\quad \text{and} \quad
\|\widehat C^\ell(\Delta)\|_{\mathrm{op}}.
$$

### 5.2 Exponential fit

Fit single-exponential decay:

$$
\|\widehat C^\ell(\Delta)\|
\approx
A_\ell e^{-\Delta/\xi_\ell}.
$$

Taking logs:

$$
\log \|\widehat C^\ell(\Delta)\|
\approx
\log A_\ell-\Delta/\xi_\ell.
$$

Fit the slope by least squares over a reasonable range $\Delta\in[\Delta_{\min},\Delta_{\max}]$.

Also fit multi-exponential decay:

$$
C^\ell(\Delta)
\approx
\sum_{\mu=1}^{M_\ell}
B_\mu^\ell \lambda_\mu^\Delta.
$$

Practical simplified version: fit scalar norms to a sum of exponentials,

$$
c^\ell(\Delta)
\approx
\sum_{\mu=1}^{M}
a_\mu e^{-\Delta/\xi_\mu}.
$$

This is less theoretically exact than matrix-valued fitting but good as a first diagnostic.

### 5.3 Required diagnostic outputs

For each model and layer:

- Plot $\log \|\widehat C^\ell(\Delta)\|_F$ vs $\Delta$.
- Plot $\log \|\widehat C^\ell(\Delta)\|_{\mathrm{op}}$ vs $\Delta$.
- Plot singular values of $C^\ell(\Delta)$ for selected $\Delta$.
- Report fitted $\xi_\ell$.
- Report fit quality, e.g. $R^2$ or held-out residual error.
- Estimate effective mode count $M_\ell$.
- Estimate predicted useful bond dimension:

$$
D_\mathrm{pred}\sim \sqrt{M_\ell}.
$$

Create a table:

| model | layer $\ell$ | feature map $\phi$ | $p$ | fit $\xi_\ell$ | single-exp fit quality | estimated $M_\ell$ | predicted $D\sim\sqrt{M_\ell}$ |
|---|---:|---|---:|---:|---:|---:|---:|

### 5.4 Interpretation rules

If correlations are approximately exponential and low-mode:

- MPS/TN is a plausible inductive bias.
- Proceed to MPS completion experiments.

If correlations are power-law or have no clean decay:

- A simple MPS may be a poor inductive bias.
- Consider tree TN, MERA-like structure, attention pooling, or longer-range virtual shortcuts.
- Still run baselines, but interpret negative results seriously.

If correlations are exponential only in some layers:

- Focus MPS experiments on those layers.
- Use non-exponential layers as negative controls.

---

## 6. Model families to implement

Implement the following in increasing complexity. Keep all models in PyTorch.

### B0: single-site linear FutureLens

Input:

$$
r_t^\ell.
$$

Prediction:

$$
\widehat r_{t+s}^{L}=A_s r_t^\ell+b_s.
$$

Use separate heads per horizon $s=1,\ldots,n$, or a shared trunk plus horizon heads.

Loss:

$$
\sum_{s=1}^{n}\|\widehat r_{t+s}^L-r_{t+s}^L\|_2^2.
$$

Also compute token-level KL and CE by decoding predicted residuals.

Purpose: reproduce FutureLens-style baseline.

### B1: multi-site linear probe

Input:

$$
[r_{t-m+1}^\ell;\ldots;r_t^\ell]\in\mathbb R^{m d_{\mathrm{model}}}.
$$

Prediction:

$$
\widehat r_{t+s}^{L}=A_s [r_{t-m+1}^\ell;\ldots;r_t^\ell]+b_s.
$$

Purpose: controls for "more context helps."

### B2: multi-site MLP

Input is the same concatenated residual window, or projected features:

$$
[v_{t-m+1}^\ell;\ldots;v_t^\ell]\in \mathbb R^{mp}.
$$

Use a small MLP. Match parameter counts with MPS where possible.

Purpose: controls for generic nonlinear expressivity.

### B3: small attention-pooling probe

Input sequence:

$$
v_{t-m+1}^\ell,\ldots,v_t^\ell.
$$

Use a tiny transformer/attention block or pooling attention to produce horizon predictions.

Purpose: controls for learned position mixing.

### B4: MPS readout over observed sites

This is the simplest tensor-network model.

Input observed sites:

$$
v_{-m+1},\ldots,v_0.
$$

Local contraction:

$$
M_j=A_j(v_j)=\sum_{a=1}^{p}v_{j,a}A_j^a.
$$

MPS hidden state:

$$
h_\theta
=
\ell_{\mathrm{bd}}^\top
M_{-m+1}\cdots M_0.
$$

Then use heads to predict future residuals:

$$
\widehat r_{t+s}^L=g_s(h_\theta).
$$

This is not the full "future sites are physical sites" completion story, but it is the fastest MPS prototype.

### B5: masked MPS completion

This is closer to the theory.

Build a chain of $m+n$ sites. For observed positions:

$$
u_j=v_j,\qquad j\le 0.
$$

For future positions:

$$
u_j=v_{\mathrm{mask},j},\qquad j>0,
$$

where $v_{\mathrm{mask},j}$ is a learned mask vector, optionally horizon-specific.

Run an MPS over all $m+n$ sites.

At each future site, produce a local hidden representation and decode:

$$
\widehat r_{t+s}^{L}=g_s(h_s).
$$

Loss:

$$
\mathcal L_{\mathrm{res}}
=
\sum_{s=1}^{n}
\|\widehat r_{t+s}^L-r_{t+s}^L\|_2^2.
$$

Purpose: implements "condition on first $m$ physical sites and complete next $n$ physical sites."

### B6: discrete conditional generative MPS

This is optional and should only be attempted after B4/B5.

Quantize residual features into codebook indices:

$$
z_j=Q(\phi(r_j^\ell))\in\{1,\ldots,K\}.
$$

Train a Born-style MPS model:

$$
P_\theta(z_{-m+1},\ldots,z_n)
=
\frac{|\ell^\top A^{z_{-m+1}}\cdots A^{z_n}r|^2}{Z}.
$$

Then condition:

$$
P_\theta(z_F\mid z_O)
=
\frac{
P_\theta(z_O,z_F)
}{
\sum_{z_F'}P_\theta(z_O,z_F')
}.
$$

Purpose: most faithful to many-body "clamped observed sites, sample future sites" language. But it is likely too much for the first pass.

---

## 7. Losses and metrics

### 7.1 Residual MSE

For predicted future residuals:

$$
\mathrm{MSE}_s
=
\mathbb E
\left[
\|\widehat r_{t+s}^{L}-r_{t+s}^{L}\|_2^2
\right].
$$

Also compute normalized MSE:

$$
\mathrm{NMSE}_s
=
\frac{
\mathbb E[\|\widehat r_{t+s}^{L}-r_{t+s}^{L}\|_2^2]
}{
\mathbb E[\|r_{t+s}^{L}-\mathbb E r_{t+s}^{L}\|_2^2]
}.
$$

### 7.2 Cosine similarity

$$
\mathrm{Cos}_s
=
\mathbb E
\left[
\frac{
\langle \widehat r_{t+s}^{L},r_{t+s}^{L}\rangle
}{
\|\widehat r_{t+s}^{L}\|_2\|r_{t+s}^{L}\|_2
}
\right].
$$

### 7.3 Teacher KL

Decode predicted residuals:

$$
q_{\theta,s}(\cdot\mid r_{t-m+1:t}^{\ell})
=
\operatorname{softmax}
\left(
W_U\operatorname{LN}(\widehat r_{t+s}^{L})+b_U
\right).
$$

Teacher distribution:

$$
p_{\Theta,s}(\cdot)
=
p_\Theta(\cdot\mid x_{\le t+s})
=
\operatorname{softmax}
\left(
W_U\operatorname{LN}(r_{t+s}^{L})+b_U
\right).
$$

Metric:

$$
\mathrm{KL}_s
=
\mathbb E
\left[
\operatorname{KL}
\left(
p_{\Theta,s}
\middle\|
q_{\theta,s}
\right)
\right].
$$

### 7.4 Cross entropy on realized token

$$
\mathrm{CE}_s
=
\mathbb E
\left[
-\log q_{\theta,s}(x_{t+s+1})
\right].
$$

### 7.5 Top-k agreement with teacher

Compare top-$k$ tokens under $q_{\theta,s}$ and teacher $p_{\Theta,s}$:

- top-1 agreement
- top-5 overlap
- Jensen-Shannon divergence
- rank of true next token under $q_{\theta,s}$

### 7.6 Scaling diagnostics

For each model family, plot:

- loss vs horizon $s$
- loss vs observed window length $m$
- loss vs bond dimension $D$
- loss vs physical dimension $p$
- loss vs parameter count
- loss vs estimated mode count $M_\ell$
- best $D$ vs predicted $D\sim\sqrt{M_\ell}$

The most important scientific plots are:

$$
\mathrm{KL}_s \text{ vs } s
$$

and

$$
\mathrm{KL}_s \text{ vs } D
$$

with baselines matched by parameter count.

---

## 8. Experimental phases

### Phase 0: repository setup

Create a clean research repo.

Suggested structure:

```text
tn-futurelens/
  README.md
  briefing.md
  pyproject.toml
  configs/
    data/
    models/
    probes/
    experiments/
  src/
    tn_futurelens/
      __init__.py
      data/
        token_datasets.py
        activation_cache.py
        windows.py
      models/
        phi.py
        baselines.py
        mps.py
        masked_mps.py
      analysis/
        correlations.py
        exp_fits.py
        transfer_modes.py
      training/
        losses.py
        train.py
        eval.py
      utils/
        config.py
        logging.py
        distributed.py
        seed.py
  scripts/
    cache_residuals.py
    compute_correlations.py
    train_probe.py
    eval_probe.py
    make_figures.py
  notebooks/
    00_sanity_checks.ipynb
    01_correlation_diagnostics.ipynb
    02_baseline_results.ipynb
    03_mps_scaling.ipynb
  results/
    figures/
    tables/
    runs/
  tests/
    test_mps_shapes.py
    test_transfer_matrix.py
    test_windowing.py
```

Use `wandb` or TensorBoard for logging. Also write every result to disk as JSON/CSV so the project is reproducible without relying on a dashboard.

### Phase 1: activation extraction

Use TransformerLens or HuggingFace hooks.

Initial models:

1. GPT-2 small: cheap sanity check.
2. GPT-2 medium or Pythia-410M: medium-scale validation.
3. Optional later: GPT-J-6B to match FutureLens more closely.

Datasets:

1. OpenWebText or a small sampled web-text corpus.
2. WikiText-103 for a cleaner sanity dataset.
3. Optional synthetic data with known finite-correlation structure.

Extract:

- source residuals $r_i^\ell$ for selected layers $\ell$
- target final residuals $r_{i+s}^{L}$ for $s=1,\ldots,n$
- token IDs $x_{i+s+1}$
- teacher logits or top-k teacher logits if full vocab logits are too large

Important storage rule:

Do not store everything blindly. Activation storage can explode. Prefer chunked storage by model/layer/split.

Use `.pt`, `.safetensors`, HDF5, or Zarr. Zarr is good for chunked arrays. PyTorch `.pt` shards are simpler.

Store metadata:

- model name
- layer
- dataset
- token range
- sequence length
- dtype
- exact hook name
- tokenizer
- random seed

### Phase 2: correlation diagnostics

Before training any MPS, compute correlations.

For each layer $\ell$ and feature map $\phi$:

1. Load residuals.
2. Fit or apply $\phi$.
3. Compute $C^\ell(\Delta)$ for $\Delta=0,\ldots,\Delta_{\max}$.
4. Compute scalar summaries.
5. Fit exponential decay.
6. Estimate $M_\ell$.
7. Save figures and tables.

Default choices:

- $\Delta_{\max}=64$ or 128.
- $p=64$ for PCA/whitened feature diagnostics.
- sample at least $10^5$ token positions for stable estimates if possible.

Be careful with document boundaries. Do not compute correlations across unrelated documents unless intentionally studying packed-stream behavior.

### Phase 3: FutureLens baselines

Train B0, B1, B2, B3.

Default setup:

- source layers: $\ell\in\{0,2,4,6,8,10,12\}$ for GPT-2 small
- horizons: $s\in\{1,2,3,4,5,8,12,16\}$
- observed lengths: $m\in\{1,2,4,8,16\}$
- source feature maps: identity, PCA-whitened $p=64$, learned $p=64$

Deliverables:

- horizon degradation plots
- parameter-matched baseline table
- layer-wise heatmap of KL or NMSE

### Phase 4: MPS readout

Train B4.

Hyperparameters:

- $D\in\{2,4,8,16,32,64\}$
- $p\in\{32,64,128,256\}$
- $m\in\{2,4,8,16\}$
- $n\in\{1,2,4,8,16\}$

Start small:

- GPT-2 small
- one layer $\ell$ with strong exponential correlation fit
- one layer $\ell$ with weak fit as a control
- $m=8$
- $n=4$
- $p=64$
- $D\in\{2,4,8,16\}$

Main question:

Does increasing $D$ improve performance in a way consistent with the measured mode count?

### Phase 5: masked MPS completion

Train B5.

The chain has length $m+n$. Observed sites get feature vectors. Future sites get learned mask vectors.

Implementation detail: the MPS should expose intermediate left/right environments so future-site predictions can use local context. For site $j$, compute a left environment from observed previous sites and a right/environment from mask/future sites if using bidirectional MPS-style completion. Simpler first version: left-to-right MPS with future mask positions; each future position has a hidden state from cumulative contraction.

Output heads:

$$
\widehat r_{t+s}^L = W_{\mathrm{out},s} h_s+b_{\mathrm{out},s}.
$$

Compare to B4. B5 is closer to theory but may be harder to optimize.

### Phase 6: stronger theoretical diagnostics

After B4/B5 work, compute learned transfer matrix spectra for trained MPS models.

For a translation-invariant MPS, compute

$$
E=\sum_{a=1}^{p}A^a\otimes \overline{A^a}.
$$

Find eigenvalues $\lambda_\mu$. Plot:

- $|\lambda_\mu|$ sorted
- implied correlation lengths $\xi_\mu=-1/\log|\lambda_\mu|$
- compare learned $\xi_\mu$ to empirical residual-stream correlation lengths

This is a key physics check. If the model is actually using the transfer-matrix mechanism, the learned transfer spectrum should reflect the empirical correlation structure.

---

## 9. Implementation details

### 9.1 MPS layer

Implement a PyTorch module:

```python
class MPSReadout(nn.Module):
    def __init__(self, p: int, D: int, N: int, translation_invariant: bool = False):
        ...
    def forward(self, v: Tensor) -> Tensor:
        # v shape: [batch, N, p]
        # returns hidden representation, scalar score, or environments
```

Two versions:

1. Non-translation-invariant: separate $A_j^a$ for each site.
2. Translation-invariant: shared $A^a$ across sites.

For a finite window with position-specific behavior, non-translation-invariant may perform better. For transfer-matrix theory, translation-invariant is cleaner. Implement both.

### 9.2 Stable contraction

For each batch sample and site:

$$
M_j = \sum_a v_{j,a}A_j^a.
$$

Naive product:

$$
H_j = H_{j-1}M_j.
$$

Potential issue: products may explode/vanish. Use one or more:

- normalize hidden matrix/vector after each site and track log-norm
- use canonical initialization
- use residual/skip connection in the readout
- use small initialization for $A$
- constrain or regularize spectral radius

Start with simple normalization:

```python
h = h @ M_j
scale = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
h = h / scale
log_scale += scale.log()
```

### 9.3 Output representation

MPS contraction can output:

- scalar score
- vector hidden state
- flattened $D\times D$ environment
- left/right environment per future site

For residual prediction, output a vector hidden representation. A practical choice is to maintain a virtual vector $h_j\in\mathbb R^D$ and decode from it. Another option is to flatten a $D\times D$ environment to dimension $D^2$ and decode from that.

The latter better matches transfer-matrix theory:

$$
h_j \in \mathbb R^{D^2}.
$$

Then

$$
\widehat r_{t+s}^{L}=W_s h_j+b_s.
$$

### 9.4 Feature map module

Implement:

```python
class IdentityPhi(nn.Module)
class PCAPhi(nn.Module)
class LearnedLinearPhi(nn.Module)
class MLPFeaturePhi(nn.Module)  # optional later
```

For PCAPhi, save fitted mean, eigenvectors, eigenvalues.

For LearnedLinearPhi, optionally initialize from PCA.

### 9.5 Parameter matching

For each model, compute trainable parameter count.

MPS parameter count roughly:

Non-translation-invariant:

$$
N p D^2 + \text{heads}.
$$

Translation-invariant:

$$
pD^2 + \text{heads}.
$$

Feature map:

$$
p d_{\mathrm{model}} + p.
$$

When comparing MPS to MLP, choose MLP hidden sizes so total trainable parameters are close.

### 9.6 Avoid leakage

When predicting future residuals $r_{t+s}^L$, ensure the input only includes positions $\le t$ for predictive models.

For correlation diagnostics, it is fine to inspect full trajectories because that is descriptive, not predictive.

For masked completion, the future sites must receive only mask vectors, not true future residuals, during prediction. During training, future residuals appear only in the loss.

### 9.7 Teacher logits storage

Full vocabulary logits can be large. Options:

1. Store teacher final residuals and compute logits on the fly.
2. Store top-$k$ teacher logits.
3. Store teacher log-probs only for evaluation batches.
4. Start with residual MSE/NMSE and add KL later.

For GPT-2 small, full logits may be manageable for small datasets. For larger models, do not store full logits for every token/horizon unless necessary.

---

## 10. Compute and storage plan

### Hardware assumption

User may provide:

- 2x NVIDIA A40 GPUs
- each A40 has 48 GB VRAM
- approximately 500 GB persistent volume disk

This is enough for the first serious version if we are disciplined.

### Recommended model scaling

Start with GPT-2 small. Then GPT-2 medium / Pythia-410M. Only then try GPT-J-6B or larger.

GPT-2 small is enough to debug:

- activation caching
- correlation diagnostics
- baselines
- MPS implementation
- plotting
- parameter matching

Do not start with GPT-J-6B.

### Storage estimates

If storing residuals in fp16, storage for selected layers is approximately:

$$
\text{bytes}
\approx
N_{\mathrm{tokens}}
\times
N_{\mathrm{layers\ stored}}
\times
d_{\mathrm{model}}
\times
2.
$$

Example: GPT-2 small, $d_{\mathrm{model}}=768$, 12 layers, 10 million token positions:

$$
10^7 \times 12 \times 768 \times 2
\approx
184\text{ GB}.
$$

This is before storing final residual targets, token IDs, metadata, and train/val/test splits.

With 500 GB disk, feasible options:

- GPT-2 small full-layer cache for several million tokens.
- GPT-2 medium selected-layer cache.
- Larger model selected layers only.

Recommended first cache:

- GPT-2 small
- layers $\ell\in\{0,2,4,6,8,10,12\}$
- final layer $L$ targets
- 2--5 million token positions
- fp16 residuals
- token IDs as int32/int64

For larger models, cache only selected source layers and final targets.

### GPU memory

MPS probes are small. The frozen transformer forward pass dominates memory during activation extraction. For GPT-2 small/medium, 2x A40 is more than enough. For GPT-J-6B, use fp16/bfloat16 and maybe one GPU for model inference plus CPU/disk streaming; two A40s should be workable but the pipeline must be designed carefully.

### Practical recommendation

Use 2x A40 as follows:

- GPU 0: transformer activation extraction
- GPU 1: probe training or parallel extraction shard
- disk: chunked activation cache
- CPU RAM: preprocessing, PCA fitting, correlation accumulation

Do not keep massive activation tensors in GPU memory. Stream batches and write shards.

---

## 11. Scientific success criteria

### Strong positive evidence

We want to see all or most of:

1. Residual-stream correlations show approximate exponential decay in some layers.
2. A small number of exponential modes fits the measured correlations.
3. MPS performance improves with $D$.
4. The useful $D$ scale is consistent with $D\sim\sqrt{M_\ell}$.
5. MPS beats parameter-matched multi-site linear/MLP/attention baselines at medium horizons.
6. Learned MPS transfer spectra resemble empirical correlation spectra.
7. MPS helps more in layers where the correlation diagnostics predict it should.

### Weak positive evidence

MPS beats single-site FutureLens but not multi-site MLP/attention. This only shows that more context or nonlinearity helps.

### Negative evidence

1. Correlations are not exponential.
2. Estimated mode count $M_\ell$ is very large.
3. MPS needs huge $D$ to match baselines.
4. Attention-pooling dominates MPS at matched parameter count.
5. Transfer spectra do not relate to empirical correlation lengths.

This is still scientifically useful: it may imply transformer residual trajectories are not finite-correlation-length one-dimensional systems in the relevant feature basis.

---

## 12. First concrete run plan

### Run 1: GPT-2 small, correlation diagnostics

Configuration:

```yaml
model: gpt2
dataset: wikitext103_or_openwebtext_sample
sequence_length: 256
num_tokens: 2_000_000
layers: [0, 2, 4, 6, 8, 10, 12]
feature_maps:
  - identity
  - pca_whitened_p64
deltas: 0..64
```

Outputs:

- `results/tables/correlation_fits_gpt2.csv`
- `results/figures/corr_decay_gpt2_layer*.png`
- `results/figures/corr_svals_gpt2_layer*.png`

### Run 2: GPT-2 small, baselines

Configuration:

```yaml
model: gpt2
source_layers: [4, 8, 10]
m_values: [1, 2, 4, 8, 16]
horizons: [1, 2, 3, 4, 5, 8, 12]
baselines: [single_linear, multisite_linear, mlp]
target: final_residual
loss: residual_mse
metrics: [nmse, cosine, teacher_kl_if_available]
```

Outputs:

- horizon degradation plot
- layer-by-horizon heatmap
- parameter-matched baseline table

### Run 3: GPT-2 small, MPS readout

Configuration:

```yaml
model: gpt2
source_layers: [best_exp_fit_layer, weak_exp_fit_control_layer]
m: 8
n: 4
p_values: [32, 64, 128]
D_values: [2, 4, 8, 16, 32]
feature_maps: [pca_whitened, learned_linear]
mps:
  translation_invariant: [false, true]
  normalize_contractions: true
```

Outputs:

- performance vs $D$
- performance vs $p$
- performance vs parameter count
- comparison against baselines
- learned transfer spectrum if translation-invariant

### Run 4: masked MPS completion

Configuration:

```yaml
model: gpt2
source_layer: best_exp_fit_layer
m: 8
n_values: [2, 4, 8]
p: 64
D_values: [4, 8, 16, 32]
feature_map: learned_linear
model_family: masked_mps_completion
```

Outputs:

- future-site completion losses
- horizon degradation curves
- comparison to MPS readout and MLP

---

## 13. Coding standards

1. Every experiment must have a config file.
2. Every run must save:
   - config
   - git commit hash if available
   - random seed
   - model name
   - dataset info
   - metrics JSON
   - parameter count
3. Every plot must be reproducible from saved CSV/JSON.
4. Do not hard-code model dimensions.
5. All tensor shapes must be documented in comments.
6. Add unit tests for:
   - window indexing
   - off-by-one residual/token target convention
   - MPS contraction shapes
   - transfer matrix shape $D^2\times D^2$
   - correlation function estimates on synthetic data
7. Add synthetic tests before using real transformer activations.

---

## 14. Synthetic data tests

Before transformer data, build synthetic processes where the answer is known.

### Synthetic process 1: AR(1)

Generate

$$
v_{i+1}=\rho v_i+\epsilon_i.
$$

Then

$$
C(\Delta)\propto \rho^\Delta.
$$

Expected: single exponential decay.

### Synthetic process 2: sum of AR modes

Generate

$$
v_i=\sum_{\mu=1}^{M}u_\mu h_{i,\mu},
$$

where

$$
h_{i+1,\mu}=\rho_\mu h_{i,\mu}+\epsilon_{i,\mu}.
$$

Expected:

$$
C(\Delta)\approx \sum_{\mu=1}^{M}B_\mu \rho_\mu^\Delta.
$$

Use this to verify whether estimated $M$ and predicted $D\sim\sqrt M$ behave sensibly.

### Synthetic process 3: long-range/power-law process

Generate a process with non-exponential correlations. Expected: MPS with small $D$ should struggle relative to baselines.

These tests ensure the correlation-fitting code and MPS implementation actually reflect the theory.

---

## 15. Reporting style

Each experiment report should include:

1. **Question:** What theoretical question is this experiment answering?
2. **Observable:** What quantity was measured?
3. **Prediction:** What does the MPS/finite-correlation-length theory predict?
4. **Controls:** What baselines rule out trivial explanations?
5. **Result:** What happened?
6. **Interpretation:** Does this support or weaken the transfer-matrix hypothesis?
7. **Next step:** What should be done based on the result?

Example:

> Question: Do layer-8 GPT-2 residual features have finite correlation length?  
> Observable: $\|\widehat C^\ell(\Delta)\|_{\mathrm{op}}$.  
> Prediction: approximately linear decay on a log plot if exponential.  
> Result: layer 8 shows $R^2=0.94$ for a single exponential fit over $\Delta=4,\ldots,32$.  
> Interpretation: layer 8 is a good candidate for MPS completion.  
> Next step: train MPS readout at layer 8 and compare to layer 2 as a weaker-correlation control.

---

## 16. What not to do

Do not:

- Start with a huge model.
- Train an MPS before measuring residual correlations.
- Compare MPS only against single-site FutureLens.
- Use future residuals as inputs in a predictive experiment.
- Store full logits for every token/horizon unless storage is budgeted.
- Treat a performance improvement as evidence for the transfer-matrix mechanism without checking learned transfer spectra and parameter-matched baselines.
- Hide failed experiments. Negative results are theoretically informative.

---

## 17. Minimal deliverables for the first milestone

The first milestone is successful when the repo contains:

1. Working activation extraction for GPT-2 small.
2. Cached residual stream windows for at least one dataset.
3. Correlation diagnostic code.
4. Exponential fit tables.
5. Single-state linear FutureLens baseline.
6. Multi-site linear baseline.
7. MPS readout baseline.
8. One report tying the results back to:
   - transformer residual-stream theory
   - FutureLens
   - MPS transfer-matrix correlations
   - the finite-correlation-length hypothesis

The first milestone does not require beating all baselines. It requires a clean test of the hypothesis.

---

## 18. Final guiding principle

The central claim is not "tensor networks are cool."

The central claim is:

$$
\text{Transformer residual trajectories may have finite-correlation-length structure.}
$$

If true, then:

$$
\text{MPS transfer matrices are the right language for modeling that structure.}
$$

All implementation choices should be justified by this claim. All results should be interpreted through it.

### aniket's extra comments
you should have a docs/ folder with folders for each experiment, each experiment folder should have a plan.md and summary.md. the latter should have inline plots and read like a summary i can hand off to dmitry and he can get a quick glance at the state of the results of that particular experiment

`snake_case` everything! (besides class names ofc)

important note for the agent: intermittently check this briefing document! you should not read it once and never again, its a reference as well as a briefing.
