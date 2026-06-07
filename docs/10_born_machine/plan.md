# Experiment 10 — Born-machine MPS: discrete conditional completion · Plan

**Motivation (briefing B6, next-step #2).** The most theory-faithful test of "clamp the
observed sites, complete the future": a genuine *generative* tensor network, not a
readout. Does a Born MPS over quantized residual symbols capture the sequence
distribution and condition usefully?

## Method
1. PCA-whiten residuals (p=64); k-means codebook (K=256) → quantize each position to a
   symbol $z=Q(\phi(r))$.
2. Build length-$N=m+n=12$ chains. Train a Born MPS $P(z)=|\Psi(z)|^2/Z$ by NLL
   (Z via the double-layer transfer matrix; log-norm stabilised).
3. **Conditional completion:** given observed $z_{1:m}$, predict the next symbol via the
   exact Born conditional $P(z_{m+1}=k\mid z_{1:m})\propto v_k^\top R\,v_k$.
4. Compare next-symbol accuracy to unigram, bigram, and a discriminative MLP classifier
   on the same quantized task.

Validated first on a sticky Markov chain (Born recovers the optimal conditional).

## Caveat
Quantizing a 768-dim residual to one of K=256 symbols is lossy; this tests the
*generative completion mechanism* on a coarse symbolic abstraction, not full-fidelity
prediction.

## Reproduce
```bash
python scripts/exp10_born.py --layer 6  --device cuda:0
python scripts/exp10_born.py --layer 12 --device cuda:0
```
