# Literature Review & Theory Notes — Tensor Network FutureLens

**Author:** coding agent (Claude), session 1
**Purpose:** synthesized, citable notes from a deep read of every reference in `briefing.md`, plus the technical decisions and gotchas surfaced while researching. This is a reference doc for the project owners (Aniket, Dmitry) and for future agent sessions. Read `briefing.md` first; this fills in the *why* behind each cited result and the *exact* facts needed to implement.

Math uses `$...$` / `$$...$$` (per briefing convention).

---

## 0. The one-paragraph thesis (and what would make it true)

FutureLens (Pal et al. 2023) showed a **single** transformer hidden state already encodes information about *several* upcoming tokens. This project asks a sharper, physics-flavored question: treat the residual stream **at a fixed layer $\ell$, along token position**, as a 1D quantum-many-body chain

$$
\mathcal R_t^{\ell;m,n}=\big(r_{t-m+1}^{\ell},\ldots,r_t^\ell,\;r_{t+1}^{\ell},\ldots,r_{t+n}^{\ell}\big),
$$

and ask whether that chain has **finite correlation length**. If it does, then a **matrix product state (MPS)** is the *right* inductive bias for completing the future sites, because a bond-$D$ MPS exactly describes a system whose connected two-point function is a sum of $\le D^2-1$ exponentials. The headline quantitative prediction is

$$
\underbrace{M_\ell}_{\text{# empirical correlation modes at layer }\ell}\;\Longrightarrow\;D_{\text{useful}}\sim\sqrt{M_\ell}.
$$

A **strong** positive result is *not* "MPS beats single-state FutureLens" (that only shows more context helps). It is: **MPS beats parameter-matched multi-site linear/MLP/attention baselines, at the horizons and layers where the measured residual correlations are well-fit by few exponential modes, and the trained MPS's transfer-matrix spectrum reproduces the empirically measured correlation lengths.** If correlations are power-law instead, a fixed-$D$ MPS *should* fail to scale — and that negative result is equally informative (it says the residual trajectory is not a finite-correlation-length 1D system in that feature basis).

---

## 1. FutureLens — the direct predecessor

**Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** — Pal, Sun, Yuan, Wallace, Bau, CoNLL 2023. [arXiv:2311.04897](https://arxiv.org/abs/2311.04897) · [code](https://github.com/KoyenaPal/future-lens) · [project](https://future.baulab.info/)

**Question.** Given one hidden state $h_t^\ell$ (one token $t$, one layer $\ell$) of **GPT-J-6B** (28 layers, $d_{\text{model}}=4096$), how much does it know about tokens at positions $\ge t+2$ — i.e. beyond the trivially-decodable next token?

**Four probes + a baseline** (the abstract collapses these into "linear vs. causal"):
- **Vocab (linear):** learned linear map $h_t^\ell \to$ logits over vocab directly.
- **HS (linear):** learned `nn.Linear(4096→4096)` predicting the *future final-layer hidden state* $\hat h_{t+N}^L \approx h_{t+N}^L$, then decode with the model's **frozen** head $\text{Sequential}(\text{ln\_f}, \text{lm\_head})$. *This is exactly the residual-regression + frozen-unembed pattern this project's losses use (briefing §2.2, §7.3).*
- **Fixed-prompt causal intervention:** transplant $h_t^\ell$ into a hand-written generic context, continue decoding (nothing learned).
- **Learned-prompt causal intervention ("soft prompt"):** learn a continuous prefix so that transplanting $h_t^\ell$ makes the downstream distribution match the true future; trained with **temperature-scaled KL distillation**.
- **Bigram baseline:** corpus next-token frequencies (≈20% at $t+1$, a floor).

**Results (Precision@1):** next token ($\approx t+1$) ≈ 97% for all trained methods; two-ahead the **learned soft prompt hits 48.4%** vs ~29% for the linear methods, and its surprisal is far lower (4.5 vs ~14). 

**Takeaways that shape this project:**
1. A single state carries a "bundle" of future tokens — *future info is real and present*.
2. **Linear decoding degrades fast** beyond $t+1$; the future info is present but **not cleanly linearly readable** — a *learned causal intervention* surfaces much more. ⚠️ **Risk flag for us:** our linear/MPS *readout* of future residuals may also struggle at long horizons; the TN bet is that **multi-site context** (not a single state) changes this. Keep the soft-prompt result in mind as the bar that "context + the right structure" must clear.
3. **Future information concentrates in *middle* layers** (≈ layer 14 of 28), unlike next-token info which peaks late. → for GPT-2 small (12 blocks) prioritize **mid layers** (~4–8) for the future-completion experiments, not just late layers.
4. **Off-by-one (confirmed everywhere):** the residual at position $i$ predicts token $i+1$. So a predicted final residual $\hat r_{t+s}^L$ decodes to a distribution over token $x_{t+s+1}$ (briefing §2.1). Their code uses an explicit `+1` target offset. **We must unit-test this** (briefing §13.6).

**Tooling note:** paper/research code = HuggingFace `transformers` + `baukit` (`TraceDict`) for capture/patching; the refreshed demo uses `nnsight`. We will use **TransformerLens** instead (cleaner residual hooks) — see §5.

---

## 2. The "lens" family — where this project sits

| Method | Decode rule | Horizon | Learned? |
|---|---|---|---|
| **Logit lens** (nostalgebraist 2020) | $\text{LN}_f(h^\ell)\,W_U$ | next token ($+1$) | no |
| **Tuned lens** (Belrose et al. 2023, [2303.08112](https://arxiv.org/abs/2303.08112)) | $\text{LN}_f(A_\ell h^\ell + b_\ell)\,W_U$ | next token ($+1$) | per-layer affine $(A_\ell,b_\ell)$, init $A_\ell=I$, KL-distilled to final dist |
| **Future lens** (Pal et al. 2023) | linear/causal | **multi-step ($+2,+3,\ldots$)** | yes |
| **TN FutureLens** (this project) | **MPS over $m$ observed sites → $n$ future sites** | multi-step | yes |

The tuned lens is the **single-site** affine version of what we generalize. Two design lessons we inherit: (a) **initialize the learned probe near identity / a residual correction** and **distill against the model's own final-layer distribution** (KL), so the probe adds no information of its own; (b) per-layer probes, because representation covariance drifts across layers.

---

## 3. Tensor networks for ML — the model machinery

**Stoudenmire & Schwab, "Supervised Learning with Quantum-Inspired Tensor Networks," NeurIPS 2016** ([1605.05775](https://arxiv.org/abs/1605.05775)).

- **Local feature map** lifts each scalar input to a physical-leg vector. Canonical $p=2$: $\phi(x)=[\cos\frac{\pi x}{2},\sin\frac{\pi x}{2}]$; order-$p$ generalization $\phi^{s}(x)=\sqrt{\binom{p-1}{s-1}}\cos(\tfrac{\pi x}{2})^{p-s}\sin(\tfrac{\pi x}{2})^{s-1}$. **Full map = tensor product** $\Phi(x)=\phi(x_1)\otimes\cdots\otimes\phi(x_N)$, a rank-1 state in $p^N$ dims.
- **Model** $f(x)=W\cdot\Phi(x)$ with the order-$N$ weight tensor $W$ stored as an **MPS / tensor train**, bond dim $D$, local cores $D\times p\times D$.
- **Optimization:** original paper uses DMRG-style two-site sweeps (merge → gradient step → SVD split + truncate to $D$ = *adaptive* bond dimension). **Modern practice (and ours): treat the MPS as a plain differentiable `nn.Module`, train with autodiff + Adam.** The paper itself notes blending SGD with sweeps is non-trivial; everyone now just backprops.
- **MNIST:** $D=120$ → **0.97% test error**. $D$ is the single expressivity knob.

> ⚠️ **Critical adaptation for us.** The $\cos/\sin$ map assumes $x\in[0,1]$; **transformer activations are unbounded reals**. The briefing's three feature maps — identity, PCA-whitening, learned-linear — are all **affine**, which sidesteps the bounded-domain problem entirely and is provably absorbable into the MPS local tensor (briefing §3.2: $A_j(W_\phi r_j)=\sum_b r_{j,b}B_j^b$ with $B_j^b=\sum_a (W_\phi)_{ab}A_j^a$). So **linear $\phi$ is the right default**; the $\cos/\sin$ physical map only becomes relevant if we add a genuinely nonlinear physical leg later, or for the Born-machine variant (which needs an *orthonormal* map or discretization — see B6 below).

**Han, Wang, Fan, Wang, Zhang, "Unsupervised Generative Modeling Using MPS," PRX 2018** ([1709.01662](https://arxiv.org/abs/1709.01662)) — the reference for **B6 (Born machine)**.
- **Born rule:** $P(v)=|\Psi(v)|^2/Z$, $\Psi(v)=\text{Tr}\,A^{(1)v_1}\cdots A^{(N)v_N}$.
- **Conditioning = clamp observed sites, sample the rest** — exact, no MCMC; via canonical form, the conditional marginals are partial MPS contractions (briefing's "clamped observed, sample future" language is literally this).
- $Z=\langle\Psi|\Psi\rangle$ is a **transfer-matrix double-layer contraction**, $O(N)$; trivial in canonical form. This exact, cheap $Z$ + gradient is the advantage over energy models.
- For continuous activations a Born machine needs an **orthonormal** local map (their Eq. 23) or a **discretized codebook** $z_j=Q(\phi(r_j))$ (briefing B6). Treat B6 as optional/after-B4/B5.

**Tooling decision.** Google's `TensorNetwork` library is **archived/dead** (2024-11-07) — do not use. For PyTorch MPS the right answer (and the briefing's) is **plain `torch.einsum` + autodiff**. Maintained fallbacks if we want a batteries-included layer: **TensorKrowch** (PyTorch-native `MPSLayer`, maintained) and **quimb** (heavy DMRG/canonicalization). **TorchMPS** is dormant but a clean reference for the Stoudenmire setup and the einsum/init patterns (see §6).

---

## 4. MPS theory — the transfer-matrix engine (the load-bearing math)

Sources: Pérez-García et al. ([quant-ph/0608197](https://arxiv.org/abs/quant-ph/0608197)), Schollwöck DMRG review ([1008.3477](https://arxiv.org/abs/1008.3477)), Zauner et al. ([1408.5140](https://arxiv.org/abs/1408.5140)), Cirac et al. RMP ([2011.12127](https://arxiv.org/abs/2011.12127)).

**Definition.** Local core $A^a_{\alpha\beta}$: physical leg $a$ (dim $p$), bond legs $\alpha,\beta$ (dim $\le D$). OBC: boundary cores are vectors ($D_0=D_N=1$). Parameter count $\approx (p-1)ND^2$ vs $p^N$ — the compression.

**Canonical forms.** Left-canonical $\sum_a A^{a\dagger}A^a=I$; right-canonical $\sum_a A^a A^{a\dagger}=I$; mixed/center form puts the Schmidt spectrum on the center bond. Why we care: orthonormal environments (stable contractions), Schmidt values read off directly (entanglement), and optimal truncation. **Gauge freedom:** $A^a\to GA^aG^{-1}$ leaves the state invariant; canonicalization = gauge fixing.

**Transfer matrix.** $E=\sum_a A^a\otimes\overline{A^a}\in\mathbb R^{D^2\times D^2}$ (real cores for our real-valued probe; $\overline{A^a}$ is a no-op, but $E$ can still have complex *eigenvalues*). Normalize leading eigenvalue $\lambda_1=1$. Spectral decomposition $E^k=\sum_{\mu=1}^{D^2}\lambda_\mu^k|r_\mu)(\ell_\mu|$.

**The central claim, made precise.** The connected two-point function is

$$
\langle O_0 P_\Delta\rangle_c=\sum_{\mu=2}^{D^2} c_\mu(O,P)\,\lambda_\mu^{\Delta-1},\qquad \lambda_\mu=e^{-\Delta/\xi_\mu}\;\text{with}\;\xi_\mu=-1/\ln|\lambda_\mu|.
$$

$E$ has $D^2$ eigenvalues; the leading one ($=1$) is the *disconnected* part and **drops out of the connected correlator**, leaving **exactly $D^2-1$ subleading exponential modes**. This is the precise version of the briefing's "$O(D^2)$ modes ⟹ $D\sim\sqrt{M}$." (Verified independently in Schollwöck Eq. 115 and Zauner Eq. 20.)

**Subtleties to respect (they affect how we fit and interpret):**
1. **Vanishing form factors:** a *specific* observable only excites modes with $c_\mu\ne0$; symmetry can null many. So $D^2-1$ is an **upper bound** on observable modes — a given correlator may show far fewer. → don't over-read a small fitted $M_\ell$.
2. **Complex eigenvalues ⟹ exponentially *decaying oscillations*** $e^{-\Delta/\xi}\cos(\phi\Delta+\delta)$ (Ornstein–Zernike). → our exponential fits should allow oscillatory/multi-exponential forms, and "log-linear decay" may have wiggles.
3. **Degeneracy** reduces the number of *distinct* correlation lengths.
4. **Jordan blocks** (non-diagonalizable $E$) give **polynomial × exponential** modes $\sim\Delta^{m-1}\lambda^\Delta$. Modulus-1 eigenvalues always have trivial Jordan structure, so normalization is safe.

**The inductive bias (why MPS ⟺ finite correlation length).** A finite-$D$ MPS has a **gapped** transfer matrix ($|\lambda_2|<1$), so at large $\Delta$ the correlator is a *pure exponential* with $\xi=-1/\ln|\lambda_2|<\infty$. **A finite-$D$ MPS therefore cannot represent power-law/critical correlations exactly** — to do so needs $|\lambda_2|\to1$, i.e. $D\to\infty$. **This is the entire bet of the project**, and it gives us a clean negative control (power-law data / power-law layers).

**Entanglement / area law.** Cutting one bond gives Schmidt rank $\le D$, so $S\le\log D$ — the 1D area law.

---

## 5. Area law ⟺ efficient MPS — the rigorous scaffolding

**Brandão & Horodecki, "Exponential Decay of Correlations Implies Area Law," CMP 2015** ([1206.2947](https://arxiv.org/abs/1206.2947)).

The logical chain we are implicitly testing:

$$
\text{gapped 1D}\;\xrightarrow{\text{Lieb–Robinson}}\;\text{exp. decay of correlations (finite }\xi)\;\xrightarrow{\text{B–H, Hastings}}\;\text{area law }(S\le\text{const})\;\xrightarrow{\text{SWVC}}\;\text{efficient MPS (small }D).
$$

- **B–H theorem (1D):** $(\xi,\ell_0)$-exponential decay $\Rightarrow$ $S(\rho_X)\le c'\,\ell_0\,e^{c\,\xi\log\xi}$ — **constant in subsystem size**, but exponential in $\xi$. Corollary: MPS approx with $D=\text{poly}(n^{\xi\log\xi},1/\delta)$.
- **Hastings 2007** ([0705.2024](https://arxiv.org/abs/0705.2024)): gapped ground states obey the area law and are MPS-approximable with $D$ *independent of $n$*.
- ⚠️ **Subtlety (Schuch–Wolf–Verstraete–Cirac, [0705.0292](https://arxiv.org/abs/0705.0292)):** a **von Neumann** area law is *not* sufficient for efficient MPS — you need bounded **Rényi** entropy (control of the entanglement-spectrum tail). "Bounded entanglement ⟹ small $D$" is the right heuristic; the rigorous version needs the Rényi tail, which B–H's correlation-decay hypothesis supplies.

**Negative control (critical case).** Power-law correlations ⟹ logarithmic entanglement $S(L)=\frac{c}{6}\log L$ ⟹ $D\gtrsim L^{c/6}$ (bond dimension must grow with system size). This is exactly the regime where a fixed-$D$ MPS probe should **fail to scale** — our designed negative control (briefing §5.4, §14 synthetic process 3).

---

## 6. Implementation cheat-sheet (TransformerLens + PyTorch MPS)

### 6.1 TransformerLens — exact API

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")   # folds LN, centers writing weights + unembed BY DEFAULT
```

- ⚠️ **Version gotcha (must pin/verify):** the legacy `HookedTransformer.from_pretrained` path folds LayerNorm and centers weights by default — *this is what makes a manual logit lens valid* (`resid @ W_U + b_U` reproduces the model). The newer **TransformerLens 3 `TransformerBridge`** preserves *raw* HF weights and does **not** fold LN — which would break the simple logit lens unless we apply full final LN (with $\gamma,\beta$) first. **Decision: pin a TransformerLens version using the legacy `HookedTransformer` API, with `fold_ln=True`.** Add a sanity test that `logit_lens(resid_post[last]) == model logits`.
- **Exact residual hook names:** `blocks.{l}.hook_resid_pre`, `blocks.{l}.hook_resid_mid`, `blocks.{l}.hook_resid_post`, `hook_embed`, `ln_final.hook_normalized`. The "final pre-unembed residual" = last block's `hook_resid_post`.
- **Caching:** `logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))`. `cache["resid_post", l]` ≡ `cache[f"blocks.{l}.hook_resid_post"]`, shape `[batch, seq, d_model]`.
- **Compute savings:** `stop_at_layer=k` (run only blocks $0..k-1$) + `names_filter` (cache only needed hooks) + `pos_slice`. Use `dtype="bfloat16"` to halve activation memory; offload cache to CPU via `run_with_cache(..., device="cpu")`.
- **Manual logit lens** (for KL/CE metrics on predicted residuals):
  ```python
  def logit_lens(resid, model):           # resid: [..., d_model]
      return model.ln_final(resid) @ model.W_U + model.b_U
  ```
- **Off-by-one (confirmed):** loss aligns `logits[:, :-1]` with `tokens[:, 1:]`; residual at pos $i$ → token $i+1$.

### 6.2 PyTorch MPS — contraction, stability, transfer spectrum

```python
# cores A: [n_sites, D, p, D] = [site, bond_l, phys, bond_r];  features v: [batch, n_sites, p]
M = torch.einsum('sdpe,bsp->bsde', A, v)            # per-site transfer matrices [batch, site, D, D]

h = left_boundary.expand(B, 1, D).clone()           # OBC: boundary VECTORS
log_norm = torch.zeros(B, device=A.device)
for j in range(n_sites):
    h = torch.einsum('bde,bef->bdf', h, M[:, j])    # == h @ M[:, j]
    nrm = h.norm(dim=(-2,-1), keepdim=True).clamp_min(1e-12)   # stability: normalize + log-norm
    h = h / nrm
    log_norm += nrm.squeeze(-1).squeeze(-1).log()
# regression readout: feed normalized h (direction) [+ log_norm as a scalar feature]
feat = h.reshape(B, -1)                              # D (after right-boundary) or D^2 (full env)
```

- **Near-identity / canonical init (the single most important stability trick):** $A^a \approx I/\sqrt{p} + \varepsilon$ so $E\approx I\otimes I$ (spectral radius ≈ 1, chain neither explodes nor vanishes):
  ```python
  A = torch.eye(D).unsqueeze(1).expand(D, p, D).clone() / p**0.5
  A = A + 1e-2 * torch.randn(D, p, D)
  ```
- **Two MPS variants** (implement both, briefing §9.1): non-translation-invariant (separate $A_j$ per site — better for finite windows) and translation-invariant (shared $A$ — required for transfer-matrix theory).
- **Transfer-matrix spectrum of a trained uniform MPS** (Phase 6 physics check):
  ```python
  E = torch.einsum('apc,bpd->abcd', A, A.conj()).reshape(D*D, D*D)   # [D^2, D^2]
  lam = torch.linalg.eigvals(E)                                       # D^2 complex eigenvalues
  lam = lam[lam.abs().argsort(descending=True)]
  xi  = -1.0 / torch.log((lam[1].abs()/lam[0].abs()).clamp_min(1e-12))  # learned correlation length
  ```
  **Compare learned $\xi$ (and the full subleading spectrum) to the empirically measured residual-stream correlation lengths** — this is *the* test that the model is actually using the transfer-matrix mechanism, not just fitting.

---

## 7. Decisions & open questions captured this session

**Decisions (defaults unless overridden):**
1. **Activation tooling:** TransformerLens, legacy `HookedTransformer` API, `fold_ln=True`; pin the version. Sanity-test logit-lens identity.
2. **Feature maps:** affine only (identity / PCA-whiten / learned-linear) — sidesteps unbounded-activation problem and is MPS-absorbable. $\cos/\sin$ / Born-machine deferred.
3. **MPS training:** plain `torch.einsum` + Adam (no DMRG sweeps); per-site normalize + log-norm; near-identity init.
4. **Logging:** `ANTHROPIC_API_KEY` and `WANDB_API_KEY` are **empty** in this env → use **TensorBoard + JSON/CSV to disk** (briefing endorses disk-first reproducibility anyway). No wandb.
5. **First model:** GPT-2 small (12 blocks, $d=768$). Prioritize **mid layers (~4–8)** for future-completion (FutureLens: future info is mid-layer). GPT-2 medium / Pythia-410M later.
6. **TDD order (briefing §13–14):** build & test on **synthetic processes first** (AR(1) → single exp; sum-of-AR → $M$ modes & $D\sim\sqrt M$; power-law → negative control) before touching transformer activations.

**Open questions for Aniket/Dmitry:**
- **Dataset:** WikiText-103 (clean, easy) vs OpenWebText sample (matches FutureLens-style web text). Briefing lists both; I'd default to **WikiText-103 for the sanity/diagnostics pass**, add an OpenWebText sample later. Document boundaries must be respected when computing position-correlations (don't correlate across unrelated docs).
- **Primary metric to optimize/report:** residual NMSE is cheapest; **teacher-KL via frozen unembed** is the most scientifically meaningful (matches FutureLens HS + tuned-lens distillation). Plan: train on residual MSE first, add KL/CE eval.
- **Scope of first milestone:** briefing §17 lists 8 deliverables (extraction → correlation diagnostics → B0/B1 → B4 MPS readout → one report). Confirm we target exactly that for milestone 1.

---

## 8. Reference index (all read this session)

- **FutureLens** — Pal et al. 2023, [arXiv:2311.04897](https://arxiv.org/abs/2311.04897), [code](https://github.com/KoyenaPal/future-lens)
- **Logit lens** — nostalgebraist 2020 (LessWrong)
- **Tuned lens** — Belrose et al. 2023, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)
- **TN supervised learning** — Stoudenmire & Schwab 2016, [arXiv:1605.05775](https://arxiv.org/abs/1605.05775)
- **Generative MPS / Born machine** — Han et al. 2018, [arXiv:1709.01662](https://arxiv.org/abs/1709.01662)
- **MPS representations** — Pérez-García et al. 2007, [arXiv:quant-ph/0608197](https://arxiv.org/abs/quant-ph/0608197)
- **DMRG/MPS review** — Schollwöck 2011, [arXiv:1008.3477](https://arxiv.org/abs/1008.3477)
- **Transfer matrices & excitations** — Zauner et al. 2015, [arXiv:1408.5140](https://arxiv.org/abs/1408.5140)
- **MPS/PEPS concepts & theorems (RMP)** — Cirac et al. 2021, [arXiv:2011.12127](https://arxiv.org/abs/2011.12127)
- **Exp. decay ⟹ area law** — Brandão & Horodecki 2015, [arXiv:1206.2947](https://arxiv.org/abs/1206.2947)
- **Area law (gapped 1D)** — Hastings 2007, [arXiv:0705.2024](https://arxiv.org/abs/0705.2024)
- **Entropy scaling & MPS simulability** — Schuch et al. 2008, [arXiv:0705.0292](https://arxiv.org/abs/0705.0292)
- **TransformerLens** — [docs](https://transformerlensorg.github.io/TransformerLens/) · [github](https://github.com/TransformerLensOrg/TransformerLens)
- **Tooling** — TensorKrowch ([arXiv:2306.08595](https://arxiv.org/abs/2306.08595)), quimb, TorchMPS; Google TensorNetwork (archived, avoid)
