"""MPS contraction shape + stability tests (briefing §13.6)."""

import pytest
import torch

from tn_futurelens.models.mps import MPSReadout


@pytest.mark.parametrize("ti", [False, True])
@pytest.mark.parametrize("readout,expected", [("vector", 4), ("env", 16), ("scalar", 1)])
def test_hidden_shapes(ti, readout, expected):
    B, p, D, N = 5, 8, 4, 6
    mps = MPSReadout(p=p, D=D, n_sites=N, translation_invariant=ti, readout=readout, seed=0)
    v = torch.randn(B, N, p)
    hidden, log_norm = mps.contract(v)
    assert hidden.shape == (B, expected)
    assert log_norm.shape == (B,)
    assert torch.isfinite(hidden).all()


@pytest.mark.parametrize("ti", [False, True])
def test_heads_shape(ti):
    B, p, D, N, d_out, n_heads = 5, 8, 4, 6, 16, 3
    mps = MPSReadout(
        p=p, D=D, n_sites=N, translation_invariant=ti,
        readout="vector", out_dim=d_out, n_heads=n_heads, seed=0,
    )
    out = mps(torch.randn(B, N, p))
    assert out.shape == (B, n_heads, d_out)


def test_long_chain_is_finite():
    """Per-site normalisation must keep a long chain finite (no overflow)."""
    B, p, D, N = 4, 6, 8, 64
    mps = MPSReadout(p=p, D=D, n_sites=N, translation_invariant=True, readout="env", seed=0)
    hidden, log_norm = mps.contract(torch.randn(B, N, p))
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(log_norm).all()


def test_gradients_flow():
    B, p, D, N = 4, 6, 4, 8
    mps = MPSReadout(p=p, D=D, n_sites=N, readout="vector", out_dim=6, n_heads=2, seed=0)
    out = mps(torch.randn(B, N, p))
    out.pow(2).mean().backward()
    grads = [g.grad for g in mps.parameters() if g.grad is not None]
    assert grads, "no gradients were produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_wrong_input_shape_raises():
    mps = MPSReadout(p=8, D=4, n_sites=6, seed=0)
    with pytest.raises(ValueError):
        mps.contract(torch.randn(5, 7, 8))  # wrong n_sites
