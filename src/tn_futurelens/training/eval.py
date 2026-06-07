r"""Shared completion-probe dataset building and evaluation (used by Exp 02-05 drivers).

Off-by-one: future site s (0-indexed) is final-layer residual r^L_{t+s+1} predicting
token x_{t+s+2}; encoded in :func:`tn_futurelens.data.windows.make_windows`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.windows import WindowSpec, make_windows


def build_completion_dataset(cache_dir: str | Path, layer: int, m: int, n: int, n_windows: int):
    """RAW observed source residuals [N,m,d] + raw final-layer target [N,n,d] + tokens [N,n]."""
    shards = sorted(Path(cache_dir).glob("shard_*.pt"))
    meta = torch.load(shards[0], map_location="cpu", weights_only=False)["meta"]
    final = meta["final_layer"]
    spec = WindowSpec(m=m, n=n)
    Xs, Ys, Ts = [], [], []
    got = 0
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        src = sh["residuals"][layer].float()
        tgt = sh["residuals"][final].float()
        toks = sh["tokens"]
        for i in range(src.shape[0]):
            w = make_windows(src[i], target=tgt[i], token_ids=toks[i], spec=spec)
            if w["anchors"].numel() == 0:
                continue
            Xs.append(w["observed"]); Ys.append(w["future"]); Ts.append(w["future_token_ids"])
            got += w["anchors"].numel()
        del sh, src, tgt
        if got >= n_windows:
            break
    return (torch.cat(Xs)[:n_windows], torch.cat(Ys)[:n_windows], torch.cat(Ts)[:n_windows], meta)


def standardize_targets(Y_raw: Tensor, n_tr: int):
    mean = Y_raw[:n_tr].mean(0, keepdim=True)
    std = Y_raw[:n_tr].std(0, keepdim=True).clamp_min(1e-6)
    return (Y_raw - mean) / std, mean, std


def nmse_per_horizon(model, Xva: Tensor, Yva_z: Tensor, device) -> list[float]:
    with torch.no_grad():
        pred = model(Xva.to(device)); Y = Yva_z.to(device)
        num = ((pred - Y) ** 2).sum(-1)
        den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(-1)
        return [(num[:, s].mean() / den[:, s].mean().clamp_min(1e-8)).item() for s in range(Y.shape[1])]


@torch.no_grad()
def token_metrics(model, Xva, Yva_raw, Tk, tgt_mean, tgt_std, gpt, device, n_eval=4000):
    """Teacher-KL and top-1 agreement, decoding predicted/true r^L via the folded-LN unembed."""
    idx = torch.randperm(Xva.shape[0])[:n_eval]
    X = Xva[idx].to(device); Yt = Yva_raw[idx].to(device); tok = Tk[idx].to(device)
    pred = model(X) * tgt_std + tgt_mean
    kl, top1 = [], []
    for s in range(Yt.shape[1]):
        tl = gpt.ln_final(Yt[:, s]) @ gpt.W_U + gpt.b_U
        sl = gpt.ln_final(pred[:, s]) @ gpt.W_U + gpt.b_U
        lt, ls = F.log_softmax(tl, -1), F.log_softmax(sl, -1)
        kl.append((lt.exp() * (lt - ls)).sum(-1).mean().item())
        top1.append((sl.argmax(-1) == tl.argmax(-1)).float().mean().item())
    return kl, top1


def summarize(name, model, Xva, Yva_z, Yva_raw, Tk, tgt_mean_d, tgt_std_d, gpt, device, extra=None):
    from ..utils.logging import count_parameters
    nmse = nmse_per_horizon(model, Xva, Yva_z, device)
    kl, top1 = token_metrics(model, Xva, Yva_raw, Tk, tgt_mean_d, tgt_std_d, gpt, device)
    rec = {"name": name, "n_params": count_parameters(model),
           "nmse_mean": round(float(np.mean(nmse)), 4), "nmse_per_h": [round(x, 4) for x in nmse],
           "kl_mean": round(float(np.mean(kl)), 4), "kl_per_h": [round(x, 4) for x in kl],
           "top1_mean": round(float(np.mean(top1)), 4), "top1_per_h": [round(x, 4) for x in top1]}
    if extra:
        rec.update(extra)
    return rec
