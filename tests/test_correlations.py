"""Correlation diagnostics on synthetic processes with known structure (briefing §14)."""

import numpy as np
import torch

from tn_futurelens.analysis.correlations import two_point_function
from tn_futurelens.analysis.exp_fits import estimate_mode_count, single_exponential_fit
from tn_futurelens.data.synthetic import (
    ar1_process,
    multi_mode_ar_process,
    power_law_process,
)


def test_ar1_single_exponential_recovery():
    r"""AR(1): whitened op-norm correlation ~ rho^Delta, xi = -1/ln rho."""
    rho = 0.8
    V = ar1_process(n_seq=300, length=400, p=4, rho=rho, seed=0)
    res = two_point_function(V, max_delta=40, whiten=True)
    c = res.operator(whitened=True).cpu().numpy()
    deltas = res.deltas.cpu().numpy()
    # Fit over the clean window Delta<=10 (at larger Delta the finite-sample noise
    # floor flattens the tail and biases xi upward).
    fit = single_exponential_fit(deltas, c, delta_min=1, delta_max=10)
    xi_true = -1.0 / np.log(rho)  # ~ 4.48
    assert fit.r2 > 0.99
    assert abs(fit.xi - xi_true) / xi_true < 0.12


def test_ar1_c0_is_identity_after_whitening():
    V = ar1_process(n_seq=300, length=300, p=5, rho=0.7, seed=1)
    res = two_point_function(V, max_delta=5, whiten=True)
    chat0 = res.Chat[0].cpu().numpy()
    np.testing.assert_allclose(chat0, np.eye(5), atol=0.05)


def test_multi_mode_count():
    r"""Sum of two AR modes: whitened trace = sum_mu rho_mu^Delta -> M=2 modes."""
    rhos = [0.5, 0.92]
    V, info = multi_mode_ar_process(
        n_seq=500, length=600, p=8, rhos=rhos, seed=0, orthonormal_mixing=True
    )
    res = two_point_function(V, max_delta=40, whiten=True)
    trace = res.trace(whitened=True).cpu().numpy()
    est = estimate_mode_count(trace, max_modes=5)
    assert est.n_modes >= 2
    # the dominant (slow) mode's correlation length should match rho=0.92
    xi_slow_true = -1.0 / np.log(0.92)
    poles = np.abs(est.best_fit.lambdas)
    xi_est = -1.0 / np.log(poles.max())
    assert abs(xi_est - xi_slow_true) / xi_slow_true < 0.25


def test_power_law_is_not_clean_single_exponential():
    r"""Negative control: power-law decay fits a single exponential worse than AR(1)."""
    rho = 0.85
    V_ar = ar1_process(n_seq=300, length=500, p=4, rho=rho, seed=3)
    V_pl, _ = power_law_process(n_seq=300, length=500, p=4, alpha=1.0, seed=3)

    def op_corr(V):
        res = two_point_function(V, max_delta=60, whiten=True)
        return res.deltas.cpu().numpy(), res.operator(whitened=True).cpu().numpy()

    d_ar, c_ar = op_corr(V_ar)
    d_pl, c_pl = op_corr(V_pl)
    # fit over the clean window where signal >> noise for both
    r2_ar = single_exponential_fit(d_ar, c_ar, 1, 20).r2
    r2_pl = single_exponential_fit(d_pl, c_pl, 1, 20).r2
    # AR(1) is a clean exponential; the power-law control is measurably worse
    assert r2_ar > 0.98
    assert r2_ar - r2_pl > 0.05
