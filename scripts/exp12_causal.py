#!/usr/bin/env python
r"""Exp 12 — TN-parameterized causal-intervention FutureLens at GPT-J scale.

Compares donor maps {single, linear, mlp, mps} in the causal-intervention setting
(learned soft prompt + donor transplant into frozen GPT-J, read the elicited future
token), plus a no-donor control and a readout baseline (decode trajectory->future
residual->unembed). Question: does the causal intervention beat the readout, and does
the TN map beat generic maps?

  python scripts/exp12_causal.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tn_futurelens.data.activation_cache import load_model
from tn_futurelens.models.intervention import SoftPromptIntervention, TrajectoryDonorMap
from tn_futurelens.models.phi import PCAPhi
from tn_futurelens.training.eval import build_completion_dataset
from tn_futurelens.utils.logging import get_logger
from tn_futurelens.utils.seed import set_seed

LOG = get_logger("exp12")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "gpt-j-6b" / "wikitext103"
OUTDIR = ROOT / "results" / "runs" / "gptj_causal"


def surprisal_top1(logits, tgt):
    lp = F.log_softmax(logits.float(), -1)
    surp = -lp.gather(-1, tgt[:, None]).squeeze(-1).mean().item()
    top1 = (logits.argmax(-1) == tgt).float().mean().item()
    return surp, top1


def eval_intervention(model, interv, traj, futtok, device, bs=128):
    interv.eval()
    n = traj.shape[0]
    H = interv.n_horizons
    surp = [[] for _ in range(H)]; acc = [[] for _ in range(H)]
    with torch.no_grad():
        for i in range(0, n, bs):
            tb = traj[i:i + bs].to(device).float()
            logits = interv(model, tb)
            for s in range(H):
                lp = F.log_softmax(logits[s].float(), -1)
                t = futtok[i:i + bs, s].to(device)
                surp[s].append(-lp.gather(-1, t[:, None]).squeeze(-1))
                acc[s].append((logits[s].argmax(-1) == t).float())
    return ([torch.cat(a).mean().item() for a in surp], [torch.cat(a).mean().item() for a in acc])


def train_intervention(model, kind, traj_tr, fut_tr, traj_va, fut_va, *, d_model, m, n,
                       device, pca, ell, P, epochs, lr, bs):
    dm = TrajectoryDonorMap(kind, d_model, m, pca=pca, seed=0)
    interv = SoftPromptIntervention(dm, d_model, n, prompt_len=P, insert_layer=ell).to(device)
    opt = torch.optim.Adam(interv.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)
    ntr = traj_tr.shape[0]
    best, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        interv.train()
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = interv(model, traj_tr[idx].to(device).float())
            loss = sum(F.cross_entropy(logits[s].float(), fut_tr[idx, s].to(device)) for s in range(n))
            loss.backward(); opt.step()
        _, acc = eval_intervention(model, interv, traj_va, fut_va, device)
        score = float(np.mean(acc))
        if score > best:
            best, best_state, since = score, copy.deepcopy(interv.state_dict()), 0
        else:
            since += 1
            if since >= 4:
                break
    interv.load_state_dict(best_state)
    return interv


def readout_baseline(model, traj_tr, futres_tr, fut_tr, traj_va, futres_va, fut_va, *,
                     d_model, m, n, device, pca, epochs, lr, bs):
    """Decode trajectory->future final residual->unembed; CE to future token (no GPT-J fwd in train)."""
    from tn_futurelens.models.mps import MPSReadout
    from tn_futurelens.models.phi import LearnedLinearPhi
    from tn_futurelens.models.probes import PhiHead
    phi = LearnedLinearPhi(d_model, 64).init_from_pca(pca).to(device)
    head = MPSReadout(p=64, D=16, n_sites=m, readout="env", out_dim=d_model, n_heads=n,
                      const_channel=True, seed=0)
    probe = PhiHead(phi, head).to(device)
    # standardize residual targets for stable training, decode unstandardized
    mean = futres_tr.reshape(-1, d_model).mean(0).to(device)
    std = futres_tr.reshape(-1, d_model).std(0).clamp_min(1e-6).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)
    ntr = traj_tr.shape[0]
    WU, bU = model.W_U, model.b_U
    for ep in range(epochs):
        perm = torch.randperm(ntr, generator=gen)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = probe(traj_tr[idx].to(device).float()) * std + mean   # [B,n,d]
            loss = 0.0
            for s in range(n):
                logits = model.ln_final(pred[:, s]) @ WU + bU
                loss = loss + F.cross_entropy(logits.float(), fut_tr[idx, s].to(device))
            loss.backward(); opt.step()
    # eval
    probe.eval(); surp = [[] for _ in range(n)]; acc = [[] for _ in range(n)]
    with torch.no_grad():
        for i in range(0, traj_va.shape[0], 128):
            pred = probe(traj_va[i:i + 128].to(device).float()) * std + mean
            for s in range(n):
                logits = model.ln_final(pred[:, s]) @ WU + bU
                t = fut_va[i:i + 128, s].to(device)
                surp[s].append(-F.log_softmax(logits.float(), -1).gather(-1, t[:, None]).squeeze(-1))
                acc[s].append((logits.argmax(-1) == t).float())
    return ([torch.cat(a).mean().item() for a in surp], [torch.cat(a).mean().item() for a in acc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=6)
    ap.add_argument("--n-windows", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kinds", nargs="+", default=["single", "linear", "mlp", "mps"])
    ap.add_argument("--tag", default="all")
    ap.add_argument("--no-readout", action="store_true")
    args = ap.parse_args()
    set_seed(0); OUTDIR.mkdir(parents=True, exist_ok=True)
    dev = args.device

    LOG.info("loading frozen GPT-J-6B (no-processing, fp16)")
    model = load_model("gpt-j-6b", device=dev, dtype=torch.float16, process_weights=False)
    d_model = model.cfg.d_model

    X, Yres, Ytok, meta = build_completion_dataset(CACHE, args.layer, args.m, args.n, args.n_windows)
    LOG.info(f"{X.shape[0]} windows; traj {tuple(X.shape)}; future tokens {tuple(Ytok.shape)}")

    # FutureLens metric: agreement with the MODEL's own future prediction (a "lens"),
    # not the realized corpus token. Teacher token at future site s = argmax of the
    # model's logits there = argmax(decode(r^L_{t+s})). Derive from cached future residuals.
    @torch.no_grad()
    def teacher_tokens(Yr):
        out = torch.empty(Yr.shape[0], Yr.shape[1], dtype=torch.long)
        for s in range(Yr.shape[1]):
            for i in range(0, Yr.shape[0], 512):
                r = Yr[i:i + 512, s].to(dev).float()
                logits = model.ln_final(r) @ model.W_U + model.b_U
                out[i:i + 512, s] = logits.argmax(-1).cpu()
        return out
    Ytch = teacher_tokens(Yres)
    LOG.info("derived teacher (model's own) future tokens")

    ntr = int(0.9 * X.shape[0])
    flat = X.reshape(-1, d_model)
    pca = PCAPhi(d_model, 64).fit(flat[torch.randperm(flat.shape[0])[:60000]]).to(dev)
    Xtr, Xva = X[:ntr], X[ntr:]
    Ytr, Yva = Ytch[:ntr], Ytch[ntr:]            # primary target: teacher (model's own) tokens
    Ytok_va = Ytok[ntr:]                           # secondary: realized corpus tokens
    Yres_tr, Yres_va = Yres[:ntr], Yres[ntr:]

    results = {}
    # unigram floor
    for s in range(args.n):
        uni = torch.bincount(Ytr[:, s], minlength=model.cfg.d_vocab).argmax()
        results.setdefault("unigram", {"top1": [], "surprisal": []})
        results["unigram"]["top1"].append(round((Yva[:, s] == uni).float().mean().item(), 4))

    for kind in args.kinds:
        interv = train_intervention(model, kind, Xtr, Ytr, Xva, Yva, d_model=d_model, m=args.m,
                                    n=args.n, device=dev, pca=pca, ell=args.layer, P=args.prompt_len,
                                    epochs=args.epochs, lr=args.lr, bs=args.bs)
        surp, acc = eval_intervention(model, interv, Xva, Yva, dev)
        results[f"interv_{kind}"] = {"top1": [round(a, 4) for a in acc], "surprisal": [round(s, 3) for s in surp]}
        LOG.info(f"  interv[{kind}] top1/h={results[f'interv_{kind}']['top1']} surp/h={results[f'interv_{kind}']['surprisal']}")

    if not args.no_readout:
        surp, acc = readout_baseline(model, Xtr, Yres_tr, Ytr, Xva, Yres_va, Yva, d_model=d_model,
                                     m=args.m, n=args.n, device=dev, pca=pca, epochs=30, lr=1.5e-3, bs=2048)
        results["readout_mps"] = {"top1": [round(a, 4) for a in acc], "surprisal": [round(s, 3) for s in surp]}
        LOG.info(f"  readout(mps) top1/h={results['readout_mps']['top1']}")

    out = {"layer": args.layer, "m": args.m, "n": args.n, "prompt_len": args.prompt_len,
           "n_windows": int(X.shape[0]), "results": results}
    json.dump(out, open(OUTDIR / f"results_{args.tag}.json", "w"), indent=2)
    LOG.info(f"wrote results_{args.tag}.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
