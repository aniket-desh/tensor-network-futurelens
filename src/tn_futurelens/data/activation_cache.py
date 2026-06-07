r"""Residual-stream activation extraction via TransformerLens (briefing §1, §9).

We use the LEGACY ``HookedTransformer.from_pretrained`` path with ``fold_ln=True``
(+ centered writing weights / unembed). This folding is exactly what makes a manual
logit lens valid: ``ln_final(resid) @ W_U + b_U`` reproduces the model's logits, so
teacher-KL / CE on predicted residuals are meaningful (see :func:`logit_lens_check`).

Layer convention (briefing r_i^l, l = 0..L for an L-block model):
  * r^0          = embeddings              = blocks.0.hook_resid_pre
  * r^l (0<l<L)  = input to block l        = blocks.l.hook_resid_pre  (= resid_post of l-1)
  * r^L          = final residual          = blocks.(L-1).hook_resid_post
The final residual r^L is the target the unembedding decodes; off-by-one: r_i^L
predicts token x_{i+1}.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def load_model(model_name: str = "gpt2", device: str = "cuda:0", dtype=torch.float32):
    """Load a frozen HookedTransformer (legacy API, LN folded)."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        device=device,
    )
    # from_pretrained doesn't always relocate every buffer to a non-default cuda
    # index; force it so multi-GPU (cuda:1) works. (For process-level pinning,
    # CUDA_VISIBLE_DEVICES + cuda:0 is also robust.)
    model = model.to(device)
    model.cfg.device = device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def resid_hook_name(layer: int, n_layers: int) -> str:
    """Hook name for briefing residual r^layer (layer in 0..n_layers)."""
    if layer < 0 or layer > n_layers:
        raise ValueError(f"layer must be in [0, {n_layers}], got {layer}")
    if layer < n_layers:
        return f"blocks.{layer}.hook_resid_pre"
    return f"blocks.{n_layers - 1}.hook_resid_post"


@torch.no_grad()
def logit_lens_check(model, tokens: Tensor, atol: float = 1e-3) -> float:
    """Sanity: ln_final + unembed of the final residual must equal model logits.

    Returns the max abs difference (should be ~0 with folded LN). Validates that our
    manual logit lens (used for teacher-KL/CE) matches the model.
    """
    tokens = tokens.to(model.cfg.device)
    logits, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n.endswith("hook_resid_post")
    )
    final = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    manual = model.ln_final(final) @ model.W_U + model.b_U
    return (manual - logits).abs().max().item()


@torch.no_grad()
def cache_residuals(
    model,
    tokens: Tensor,
    layers: list[int],
    *,
    batch_size: int = 32,
    store_dtype: torch.dtype = torch.float16,
    include_final: bool = True,
) -> dict[int, Tensor]:
    """Cache residuals for ``layers`` (+ final r^L) over ``tokens`` ``[S, T]``.

    Returns ``{layer: tensor[S, T, d_model]}`` on CPU in ``store_dtype``.
    """
    n_layers = model.cfg.n_layers
    wanted = set(layers)
    if include_final:
        wanted.add(n_layers)
    names = {l: resid_hook_name(l, n_layers) for l in wanted}
    name_set = set(names.values())

    chunks: dict[int, list[Tensor]] = {l: [] for l in wanted}
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i : i + batch_size].to(model.cfg.device)
        _, cache = model.run_with_cache(batch, names_filter=lambda n: n in name_set)
        for l, hn in names.items():
            chunks[l].append(cache[hn].to(store_dtype).cpu())
    return {l: torch.cat(v, dim=0) for l, v in chunks.items()}


def save_shard(
    out_dir: str | Path,
    shard_idx: int,
    residuals: dict[int, Tensor],
    tokens: Tensor,
    article_ids: Tensor,
    meta: dict,
) -> Path:
    """Write one shard ``shard_{idx}.pt`` with residuals + tokens + provenance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(
        {"residuals": {int(k): v for k, v in residuals.items()},
         "tokens": tokens, "article_ids": article_ids, "meta": meta},
        path,
    )
    return path


def iter_shards(cache_dir: str | Path):
    """Yield loaded shard dicts in order."""
    for path in sorted(Path(cache_dir).glob("shard_*.pt")):
        yield torch.load(path, map_location="cpu", weights_only=False)


def load_layer(cache_dir: str | Path, layer: int) -> tuple[Tensor, Tensor]:
    """Concatenate one layer's residuals (+ article_ids) across all shards."""
    res, aids = [], []
    for shard in iter_shards(cache_dir):
        res.append(shard["residuals"][layer])
        aids.append(shard["article_ids"])
    return torch.cat(res, dim=0), torch.cat(aids, dim=0)
