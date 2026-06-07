"""Window indexing + off-by-one convention tests (briefing §13.6)."""

import torch

from tn_futurelens.data.windows import (
    WindowSpec,
    make_windows,
    token_index_for_site,
    valid_anchor_range,
)


def test_token_index_off_by_one():
    # future site s of anchor t sits at position t+s; predicts token x_{t+s+1}
    assert token_index_for_site(t=10, s=1) == 12
    assert token_index_for_site(t=10, s=3) == 14


def test_valid_anchor_range():
    spec = WindowSpec(m=4, n=3)
    # observed needs t>=3; future residual sites need t<=T-1-n
    assert valid_anchor_range(seq_len=20, spec=spec) == (3, 16)
    # realized token needs one more position
    assert valid_anchor_range(seq_len=20, spec=spec, need_realized_token=True) == (3, 15)


def test_make_windows_indices():
    T, d = 30, 2
    # source[i] encodes its own index so we can verify slicing
    source = torch.arange(T).float().unsqueeze(1).repeat(1, d)  # [T, d], row i == i
    token_ids = torch.arange(T)
    spec = WindowSpec(m=4, n=3)
    w = make_windows(source, token_ids=token_ids, spec=spec)

    t0 = int(w["anchors"][0].item())
    assert t0 == 3  # first valid anchor (m-1)
    # observed sites are positions [t-3 .. t]
    obs0 = w["observed"][0][:, 0]
    assert obs0.tolist() == [t0 - 3, t0 - 2, t0 - 1, t0]
    # future residual sites are positions [t+1 .. t+3]
    fut0 = w["future"][0][:, 0]
    assert fut0.tolist() == [t0 + 1, t0 + 2, t0 + 3]
    # realized tokens: site s=1..3 -> x_{t+s+1}
    assert w["future_token_ids"][0].tolist() == [t0 + 2, t0 + 3, t0 + 4]


def test_doc_boundary_respected():
    T, d = 20, 1
    source = torch.arange(T).float().unsqueeze(1)
    # two documents: [0..9], [10..19]
    doc_ids = torch.tensor([0] * 10 + [1] * 10)
    spec = WindowSpec(m=3, n=2)
    w = make_windows(source, spec=spec, doc_ids=doc_ids)
    anchors = w["anchors"].tolist()
    # no window may straddle the boundary at index 10: span [t-2, t+2] must stay in one doc.
    for t in anchors:
        span = list(range(t - 2, t + 3))
        docs = {int(doc_ids[i]) for i in span}
        assert len(docs) == 1
    # anchor t=8 (span 6..10) crosses -> must be excluded; t=7 (span 5..9) ok
    assert 8 not in anchors
    assert 7 in anchors
