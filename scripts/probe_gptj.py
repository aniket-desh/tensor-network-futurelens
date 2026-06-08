#!/usr/bin/env python
r"""Feasibility probe for GPT-J-6B in TransformerLens (gates Exp 12)."""

from __future__ import annotations

import time

import torch

from tn_futurelens.data.activation_cache import logit_lens_check
from tn_futurelens.utils.logging import get_logger

LOG = get_logger("probe_gptj")


def main():
    dev = "cuda:0"
    LOG.info("loading gpt-j-6b (fp16)... (downloads ~weights on first run)")
    t0 = time.time()
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt-j-6b", fold_ln=True, center_writing_weights=True, center_unembed=True,
        dtype=torch.float16, device=dev,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOG.info(f"loaded in {time.time()-t0:.0f}s; n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")
    LOG.info(f"GPU mem allocated: {torch.cuda.memory_allocated(dev)/1e9:.1f} GB")

    toks = model.to_tokens("The Eiffel Tower is located in the city of")
    t0 = time.time()
    with torch.no_grad():
        logits = model(toks)
    LOG.info(f"forward {tuple(toks.shape)} in {time.time()-t0:.2f}s; top next token: "
             f"{model.to_string(logits[0,-1].argmax())!r}")

    # logit-lens sanity (folded LN -> manual decode == model logits)
    diff = logit_lens_check(model, toks)
    LOG.info(f"logit-lens sanity max|manual-model| = {diff:.3e} ({'OK' if diff < 0.05 else 'WARN'})")

    # time a batched forward at a realistic caching size
    batch = model.to_tokens(["the quick brown fox"] * 32)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        _ = model(batch)
    torch.cuda.synchronize()
    LOG.info(f"batched forward [32, {batch.shape[1]}] in {time.time()-t0:.2f}s; "
             f"peak mem {torch.cuda.max_memory_allocated(dev)/1e9:.1f} GB")
    LOG.info("PROBE OK")


if __name__ == "__main__":
    main()
