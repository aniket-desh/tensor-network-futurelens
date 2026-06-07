r"""Local residual-trajectory windows and the off-by-one token convention.

A window anchored at position ``t`` over a source layer ``l`` is

    R_t^{l;m,n} = (r_{t-m+1}^l, ..., r_t^l,  r_{t+1}^l, ..., r_{t+n}^l).

The first ``m`` sites are *observed*; the next ``n`` are *future* sites to predict.

OFF-BY-ONE (briefing §2.1, confirmed against FutureLens/TransformerLens):

    a residual at position j (final layer L) predicts token x_{j+1}.

So a predicted future final-layer residual ``r^L_{t+s}`` (future site s = 1..n,
absolute position j = t+s) corresponds to the realized token ``x_{t+s+1}``.
This module is the single source of truth for that indexing; ``token_index_for_site``
encodes it and is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry: ``m`` observed sites, ``n`` future sites."""

    m: int
    n: int

    def __post_init__(self) -> None:
        if self.m < 1 or self.n < 1:
            raise ValueError(f"need m>=1 and n>=1, got m={self.m}, n={self.n}")

    @property
    def size(self) -> int:
        """Total chain length m + n."""
        return self.m + self.n


def token_index_for_site(t: int, s: int) -> int:
    """Absolute token index realized at future site ``s`` (1-indexed) of anchor ``t``.

    Future site ``s`` sits at position ``t + s`` whose final-layer residual predicts
    token ``x_{t+s+1}``. Returns ``t + s + 1``.
    """
    if s < 1:
        raise ValueError(f"future site s is 1-indexed (s>=1), got {s}")
    return t + s + 1


def valid_anchor_range(
    seq_len: int, spec: WindowSpec, need_realized_token: bool = False
) -> tuple[int, int]:
    """Inclusive ``[t_min, t_max]`` of anchors with a fully in-bounds window.

    Observed sites need ``t - m + 1 >= 0`` -> ``t >= m - 1``.
    Future residual sites need ``t + n <= seq_len - 1`` -> ``t <= seq_len - 1 - n``.
    If ``need_realized_token`` we also need ``x_{t+n+1}`` to exist, i.e.
    ``t + n + 1 <= seq_len - 1`` -> ``t <= seq_len - 2 - n``.
    Returns ``(t_min, t_max)`` with ``t_max < t_min`` signalling "no valid anchors".
    """
    t_min = spec.m - 1
    t_max = seq_len - 1 - spec.n - (1 if need_realized_token else 0)
    return t_min, t_max


def make_windows(
    source: Tensor,
    target: Tensor | None = None,
    token_ids: Tensor | None = None,
    spec: WindowSpec | None = None,
    *,
    m: int | None = None,
    n: int | None = None,
    doc_ids: Tensor | None = None,
    stride: int = 1,
) -> dict[str, Tensor]:
    r"""Extract windows from a single sequence.

    Args:
        source: ``[T, d_src]`` residuals at the source layer (observed sites taken here).
        target: ``[T, d_tgt]`` residuals at the (usually final) target layer; future
            targets taken here. Defaults to ``source`` if None.
        token_ids: optional ``[T]`` int token ids; if given, ``future_token_ids`` are
            returned using the off-by-one (needs the realized token to exist).
        spec / (m, n): window geometry (pass either a WindowSpec or m and n).
        doc_ids: optional ``[T]`` int document ids; windows are dropped if the span
            ``[t-m+1, t+n]`` (or ``t+n+1`` when token_ids given) crosses a doc boundary.
        stride: step between consecutive anchors.

    Returns dict with:
        ``observed``    [N, m, d_src],
        ``future``      [N, n, d_tgt],
        ``anchors``     [N]  (the anchor positions t),
        ``future_token_ids`` [N, n] (only if token_ids given).
    """
    if spec is None:
        if m is None or n is None:
            raise ValueError("pass either spec= or both m= and n=")
        spec = WindowSpec(m=m, n=n)
    if source.ndim != 2:
        raise ValueError(f"source must be [T, d], got {tuple(source.shape)}")
    if target is None:
        target = source
    T = source.shape[0]
    need_tok = token_ids is not None
    t_min, t_max = valid_anchor_range(T, spec, need_realized_token=need_tok)

    anchors: list[int] = []
    for t in range(t_min, t_max + 1, stride):
        if doc_ids is not None:
            lo = t - spec.m + 1
            hi = t + spec.n + (1 if need_tok else 0)  # last index that must share the doc
            if not bool((doc_ids[lo:hi + 1] == doc_ids[t]).all()):
                continue
        anchors.append(t)

    if not anchors:
        out: dict[str, Tensor] = {
            "observed": source.new_empty((0, spec.m, source.shape[1])),
            "future": target.new_empty((0, spec.n, target.shape[1])),
            "anchors": torch.empty(0, dtype=torch.long),
        }
        if need_tok:
            out["future_token_ids"] = torch.empty(0, spec.n, dtype=torch.long)
        return out

    anchors_t = torch.tensor(anchors, dtype=torch.long)
    # observed sites: positions [t-m+1 .. t]
    obs_idx = anchors_t[:, None] + torch.arange(-spec.m + 1, 1)  # [N, m]
    # future sites: positions [t+1 .. t+n]
    fut_idx = anchors_t[:, None] + torch.arange(1, spec.n + 1)  # [N, n]

    out = {
        "observed": source[obs_idx],   # [N, m, d_src]
        "future": target[fut_idx],     # [N, n, d_tgt]
        "anchors": anchors_t,
    }
    if need_tok:
        # token realized at future site s=1..n is x_{t+s+1}
        tok_idx = anchors_t[:, None] + torch.arange(2, spec.n + 2)  # t + (s+1), s=1..n
        out["future_token_ids"] = token_ids[tok_idx]
    return out
