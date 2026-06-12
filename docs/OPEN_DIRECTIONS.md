# Open directions with positive outlook

The project's headline result is negative (no tensor-network FutureLens advantage —
see [FINDINGS.md](FINDINGS.md) and the sprint reports in
[12_mythos_sprint](12_mythos_sprint/summary.md) /
[16_matrix_product_physics_sprint](16_matrix_product_physics_sprint/summary.md)).
This file logs the threads that emerged *positive-looking* along the way: places
where the evidence points at something real and unexploited. Each entry gives the
observation that motivates it, why the outlook is positive, and a concrete first
experiment.

---

## 1. An architecture that converts the power-law bulk into prediction

**Observation (Exp 16D):** after removing the persistent subspace, residual-stream
bulk correlations decay as a power law (α ≈ 0.4–0.75, AIC winner 8/8 across layers
and block scales) — there is long-range, scale-free predictive structure that decays
slowly with distance.

**Why positive outlook:** that structure demonstrably exists and no probe tested so
far is designed for it. Our two quick prototypes (dilated conv, tree pooling) failed,
but they were small and naive — a failure of two architectures, not of the idea. The
target is concrete and falsifiable: beat a tuned MLP on per-position top-1 at
positions s ≳ 8, where every dense model decays fastest.

**First experiment:** log-spaced/multi-resolution feature banks (wavelet- or
Fourier-style positional kernels over the observed window), an SSM/long-conv probe
(H3/Hyena-style kernels are *parameterized* as power-law-ish long convolutions —
unusually good prior match), and a residual combination `linear(persistent sector) +
long-kernel(bulk)`. Evaluate per-position, lr-swept (see §7).

## 2. MERA / RG-geometry probes — now with the *correct* motivation

**Observation (Exp 15 + 16D):** block coarse-graining leaves block-ξ scale-invariant
and raises mode count; the bulk is critical-like. In many-body terms this is exactly
the regime where MPS is the wrong ansatz and MERA is the natural one (MERA was
designed for scale-invariant/critical systems; finite-D MPS gives sums of
exponentials).

**Why positive outlook:** the original project picked MPS on a physics analogy that
turned out wrong; MERA is the ansatz the *measured* physics actually selects. This is
the only tensor-network direction the data still supports.

**Caution / gate:** sprint-2's cheap multiscale baselines failed, so do not build a
full MERA first. Gate it on a cheap diagnostic: compute correlations in MERA-style
coarse-grained coordinates (disentangler-free tree first); only invest in trainable
MERA if the effective description simplifies where block-averaging (Exp 15) did not.

## 3. The power-law bulk as a physics-of-transformers study (probe-free)

**Observation (Exp 16D):** the exponent α falls with block size (0.75 → 0.37 at
layer 6) and differs across layers; the structure replicated at GPT-2 small layers
6/8 and is robust to the persistent-rank choice.

**Why positive outlook:** this is a clean, unexplained empirical regularity about
residual streams, decoupled from any probe architecture. Natural connections: the
known power-law decay of mutual information in natural text (Lin & Tegmark 2016),
Zipfian token statistics, and criticality claims about LLM representations. The
`V ≈ G (persistent) + B (scale-free bulk)` decomposition may predict things —
e.g. how in-context information is retained over distance.

**First experiment:** α(layer, model, scale) across GPT-2 small/medium/large + one
modern model; compare against the mutual-information decay of the *input text*
itself to test "the stream inherits the corpus's long-range statistics" vs
"the network builds its own."

## 4. Matrix-product random features as a kernel-theory object

**Observation (Exp 16A):** a *frozen* random near-identity non-commuting
matrix-product basis + trained linear φ + linear head exactly matches a fully trained
MPS (Δ = +0.0002); rank-2 core slices suffice; commuting bases collapse; random
*orthogonal* slices are measurably worse (−0.15%) than near-identity ones;
averaging several site orders hurts while any single arbitrary order works.

**Why positive outlook:** these are crisp, surprising regularities about a
random-features family that (to our knowledge) isn't characterized in the
random-features/kernel literature. Theory questions with experiments attached: what
kernel does the random near-identity matrix-product chain induce? Why is the
perturbative (near-identity) regime better than isotropic randomness? Why does the
effective slice-rank saturate at 2? Why does order-mixing destroy feature quality?

**First experiment:** kernel-regression view — compute the empirical NTK/feature
Gram of the frozen chain vs the bond-free product and vs random ReLU features at
matched dimension, and see which spectral property predicts the .0994/.0956 gap.

## 5. A practical niche: zero-tuning, low-data probes

**Observation (Exp 16B data-size axis):** the frozen-MPS probe beats tuned MLP and
bilinear at 10k training windows (.0971 vs .0956/.0914), with crossover ≈ 20k; it is
also within 0.1% of its own optimum across a 5× lr range (regret 0.09% vs MLP 0.51%).

**Why positive outlook:** "best probe under ≲20k examples and no tuning budget" is a
real use case — interpretability practitioners often probe with small bespoke
datasets and default hyperparameters. The claim is already supported at n=8/GPT-2
small; it just needs a focused robustness pass to be quotable.

**First experiment:** replicate the low-data win across layers, horizons and one
more model; characterize the crossover point as a function of feature dimension.

## 6. Causal intervention as the FutureLens substrate (from Exp 12)

**Observation:** intervention ≫ readout at GPT-J (top-1 0.179–0.201 vs 0.072,
replicating FutureLens), with the *single most-recent state* as the best donor.

**Why positive outlook:** orthogonal to the TN story and consistently the largest
effect size in the whole project. Any future "what does the model know about its
future" work should build on the intervention scaffold in `exp12_causal.py`
(raw-state transplant + KL distillation + longer prompts were identified but never
run — they raise absolute numbers).

## 7. Methodology export: lr-response curves as the comparison standard

**Observation (sprint 15 → 16):** a fully rigorous-looking result (4 seeds, held-out
test, cluster-bootstrap CIs, 61/64 positive comparisons) flipped sign when the lr
grid was extended one decade — because all models had shared one recipe.

**Why positive outlook:** this is a publishable cautionary example with unusually
clean before/after artifacts (all runs, seeds and CIs are in
`docs/12_mythos_sprint/tables` and `docs/16_matrix_product_physics_sprint/tables`).
A short methods note — "architecture comparisons at a shared recipe measure
recipe-fit, not capability; report response curves" — would be useful well beyond
this project.

## 8. Smaller puzzles logged for completeness

- **Per-position profiles as a metric** (sprint-1 Finding 2): bilinear dominates
  s=1 then collapses; every model family has a characteristic decay shape. Probe
  *design* targeting specific position ranges is unexplored.
- **The persistent sector G** carries most easy predictivity; nothing was ever built
  that models G explicitly and routes only the residual to a bulk-specific module
  (TASK-1 Experiment C was never run in its full hybrid form).
- **Order-mixing hurts** (sym4, −0.48%): a single fixed random order is a good
  feature basis, an average of four is not — a concrete, isolatable phenomenon for
  direction 4's theory.
