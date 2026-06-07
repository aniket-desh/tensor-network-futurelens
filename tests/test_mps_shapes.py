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


@pytest.mark.parametrize("ti", [False, True])
def test_const_channel(ti):
    """Constant channel augments p->p+1 internally; output shape unchanged; trains."""
    B, p, D, N = 5, 8, 4, 6
    mps = MPSReadout(p=p, D=D, n_sites=N, translation_invariant=ti, readout="env",
                     out_dim=7, n_heads=2, const_channel=True, seed=0)
    assert mps.p_eff == p + 1
    core = mps.core if ti else mps.cores
    assert core.shape[-2] == p + 1                      # physical leg is p+1
    out = mps(torch.randn(B, N, p))                     # input is still p
    assert out.shape == (B, 2, 7)
    out.pow(2).mean().backward()
    assert all(torch.isfinite(g.grad).all() for g in mps.parameters() if g.grad is not None)


def test_const_channel_represents_constant():
    """With zero input, const-channel MPS gives a nonzero (input-independent) output;
    a no-const MPS env-readout gives the same output for any zero input."""
    mps = MPSReadout(p=4, D=3, n_sites=5, readout="env", out_dim=2, const_channel=True, seed=1)
    out = mps(torch.zeros(3, 5, 4))
    assert out.abs().sum() > 0  # the A^0 (constant) part carries through


@pytest.mark.parametrize("const", [False, True])
def test_masked_mps_completion(const):
    from tn_futurelens.models.masked_mps import MaskedMPSCompletion
    B, p, D, m, n, d_out = 5, 8, 4, 6, 3, 16
    model = MaskedMPSCompletion(p=p, D=D, m=m, n=n, d_out=d_out, const_channel=const, seed=0)
    out = model(torch.randn(B, m, p))
    assert out.shape == (B, n, d_out)
    out.pow(2).mean().backward()
    grads = [g.grad for g in model.parameters() if g.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the learned mask must receive gradient (future sites are used)
    assert model.mask.grad is not None and model.mask.grad.abs().sum() > 0


def test_wrong_input_shape_raises():
    mps = MPSReadout(p=8, D=4, n_sites=6, seed=0)
    with pytest.raises(ValueError):
        mps.contract(torch.randn(5, 7, 8))  # wrong n_sites
