import math

import numpy as np
import pandas as pd
import pytest

from grokking.stats import cliffs_delta, holm, km_quantile, scaling_fit, summarize


def test_km_without_censoring_is_plain_quantile():
    times = np.array([10.0, 30.0, 20.0, 40.0])
    assert km_quantile(times, np.ones(4, bool), 0.5) == pytest.approx(np.median(times))


def test_km_with_censoring():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.array([True, True, False, False])
    # S(1) = 0.75, S(2) = 0.5 -> медиана 2; до S <= 0.25 кривая не доходит -> inf
    assert km_quantile(times, observed, 0.5) == 2.0
    assert km_quantile(times, observed, 0.75) == math.inf


def test_holm_known_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cliffs_delta():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_scaling_fit_recovers_known_slopes():
    rng = np.random.default_rng(0)
    rows = []
    for lr in [1e-3, 2e-3, 4e-3]:
        for rho in [1e-6, 1e-5, 1e-4]:
            for seed in range(5):
                t = 1e-2 * lr**-1.0 * rho**-0.2 * 10 ** rng.normal(0, 0.01)
                rows.append({"reg_kind": "l2", "lr": lr, "rho": rho, "seed": seed, "outcome": "grokking", "t_grok": t})
    fit = scaling_fit(pd.DataFrame(rows), "l2")
    assert fit["alpha"] == pytest.approx(1.0, abs=0.02)
    assert fit["beta"] == pytest.approx(0.2, abs=0.02)
    assert fit["p_alpha_eq_beta"] < 1e-6


def test_summarize_counts_and_censoring():
    det = pd.DataFrame({
        "reg_kind": ["l2"] * 4, "lr": [1e-3] * 4, "rho": [1e-4] * 4, "seed": range(4),
        "outcome": ["grokking", "grokking", "no_generalization", "no_generalization"],
        "t_mem": [100.0] * 4, "t_grok": [300.0, 500.0, np.nan, np.nan], "epochs": [6000] * 4,
        "t_gen95": [400.0, 600.0, np.nan, np.nan], "censored_gen95": [False, False, True, True],
        "end_test_acc": [1.0, 1.0, 0.0, 0.0], "l2_norm_end": [1.0] * 4, "frac_small_end": [0.0] * 4,
    })
    row = summarize(det).iloc[0]
    assert row.n == 4 and row.n_grokking == 2 and row.n_no_generalization == 2
    assert row.t_grok_median == 500.0  # S(300) = 0.75, S(500) = 0.5
    assert row.share_gen95 == 0.5
