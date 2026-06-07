"""Feature-map (phi) tests."""

import torch

from tn_futurelens.models.phi import LearnedLinearPhi, PCAPhi, build_phi


def test_pca_whitens():
    torch.manual_seed(0)
    d = 6
    # data with a non-trivial covariance
    L = torch.randn(d, d)
    x = torch.randn(4000, d) @ L.T + 3.0
    pca = PCAPhi(d_model=d, p=d).fit(x)
    z = pca(x)
    cov = torch.cov(z.T)
    # whitened covariance ~ identity
    assert torch.allclose(cov, torch.eye(d), atol=0.1)
    assert z.mean(0).abs().max() < 0.1


def test_learned_linear_init_from_pca_matches():
    torch.manual_seed(0)
    d, p = 8, 4
    x = torch.randn(2000, d) @ torch.randn(d, d).T
    pca = PCAPhi(d_model=d, p=p).fit(x)
    lin = LearnedLinearPhi(d_model=d, p=p).init_from_pca(pca)
    assert torch.allclose(pca(x), lin(x), atol=1e-4)


def test_build_phi_identity():
    phi = build_phi("identity", d_model=16)
    x = torch.randn(3, 16)
    assert torch.equal(phi(x), x)
    assert phi.p == 16
