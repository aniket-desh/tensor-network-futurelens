"""Exponential-fit and Prony mode-extraction tests."""

import numpy as np

from tn_futurelens.analysis.exp_fits import (
    estimate_mode_count,
    prony_modes,
    single_exponential_fit,
)


def test_single_exponential_recovers_xi():
    xi_true = 5.0
    deltas = np.arange(0, 40)
    values = 2.0 * np.exp(-deltas / xi_true)
    fit = single_exponential_fit(deltas, values, delta_min=1, delta_max=30)
    assert np.isclose(fit.xi, xi_true, atol=1e-6)
    assert fit.r2 > 0.999


def test_prony_recovers_two_modes_exactly():
    # clean deterministic sum of exponentials
    k = np.arange(0, 24)
    c = 1.5 * (0.9 ** k) + 0.5 * (0.4 ** k)
    fit = prony_modes(c, n_modes=2)
    poles = np.sort(np.real(fit.lambdas))
    np.testing.assert_allclose(poles, [0.4, 0.9], atol=1e-4)
    # amplitudes line up with the sorted poles
    order = np.argsort(np.real(fit.lambdas))
    amps = np.real(fit.amplitudes)[order]
    np.testing.assert_allclose(amps, [0.5, 1.5], atol=1e-4)
    assert fit.reconstruction_rmse < 1e-8


def test_estimate_mode_count_picks_two():
    k = np.arange(0, 30)
    c = 1.0 * (0.85 ** k) + 1.0 * (0.45 ** k)
    est = estimate_mode_count(c, max_modes=5)
    assert est.n_modes == 2
    assert est.predicted_bond_dim == int(np.ceil(np.sqrt(2)))  # = 2
