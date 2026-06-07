r"""Losses and metrics (briefing §7).

Residual-space: MSE, NMSE, cosine. Token-space (decode predicted residuals through
the FROZEN unembedding): teacher KL, realized-token CE, top-k agreement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# -- residual-space metrics ------------------------------------------------
def residual_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over the feature dim, averaged over batch (and horizons)."""
    return ((pred - target) ** 2).sum(-1).mean()


def residual_nmse(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalised MSE: ``E||pred-target||^2 / E||target - E[target]||^2``."""
    num = ((pred - target) ** 2).sum(-1).mean()
    tgt_mean = target.mean(dim=0, keepdim=True)
    den = ((target - tgt_mean) ** 2).sum(-1).mean().clamp_min(eps)
    return num / den


def cosine_similarity(pred: Tensor, target: Tensor) -> Tensor:
    return F.cosine_similarity(pred, target, dim=-1).mean()


# -- token-space decoding (logit lens on predicted residuals) --------------
def decode_residual_to_logits(
    resid: Tensor,
    W_U: Tensor,
    b_U: Tensor | None,
    ln_final=None,
) -> Tensor:
    r"""Apply final LayerNorm + unembed to a residual vector -> logits.

    Mirrors the model's own final step (valid because TransformerLens folds LN and
    centers the unembed). ``ln_final`` is a callable (e.g. ``model.ln_final``); if
    None, only the unembed is applied (caller has pre-normalised).
    """
    x = ln_final(resid) if ln_final is not None else resid
    logits = x @ W_U
    if b_U is not None:
        logits = logits + b_U
    return logits


def teacher_kl(
    pred_logits: Tensor, teacher_logits: Tensor
) -> Tensor:
    r"""``KL(teacher || student)`` averaged over batch. Both are raw logits."""
    log_q = F.log_softmax(pred_logits, dim=-1)
    log_p = F.log_softmax(teacher_logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(-1).mean()


def realized_token_ce(pred_logits: Tensor, token_ids: Tensor) -> Tensor:
    """Cross-entropy of predicted distribution against the realized token id."""
    return F.cross_entropy(pred_logits, token_ids)


def topk_agreement(pred_logits: Tensor, teacher_logits: Tensor, k: int = 1) -> Tensor:
    """Fraction of examples whose top-1 prediction is in the teacher's top-k."""
    pred_top1 = pred_logits.argmax(dim=-1, keepdim=True)
    teacher_topk = teacher_logits.topk(k, dim=-1).indices
    hit = (teacher_topk == pred_top1).any(dim=-1).float()
    return hit.mean()
