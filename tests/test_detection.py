import math

import numpy as np
import pytest

from grokking.detection import changepoint_l2, detect, stable_crossing, synthetic_curves

STEPS = np.arange(0, 6001, 10)


def brute_force_changepoint(a):
    best_k, best_sse = None, math.inf
    for k in range(1, len(a)):
        sse = ((a[:k] - a[:k].mean()) ** 2).sum() + ((a[k:] - a[k:].mean()) ** 2).sum()
        if sse < best_sse - 1e-12:
            best_k, best_sse = k, sse
    return best_k, best_sse


def test_changepoint_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(300):
        a = rng.random(rng.integers(2, 50))
        if trial % 2:
            a = np.round(a, 1)  # повторы значений проверяют выбор при равенстве SSE
        k, sse = changepoint_l2(a)
        _, sse_ref = brute_force_changepoint(a)
        assert sse == pytest.approx(sse_ref, abs=1e-9)
        alt = ((a[:k] - a[:k].mean()) ** 2).sum() + ((a[k:] - a[k:].mean()) ** 2).sum()
        assert alt == pytest.approx(sse_ref, abs=1e-9)


def test_clean_grokking_curve_found_precisely():
    rng = np.random.default_rng(1)
    train, test = synthetic_curves(STEPS, t_mem=300, t_star=1500, width=0.15, low=0.0, high=1.0, noise=0.01, rng=rng)
    d = detect(STEPS, train, test)
    assert d.outcome == "grokking"
    assert abs(math.log10(d.t_grok / 1500)) < 0.05
    assert d.t_mem < d.t_grok
    assert not d.censored and not d.censored_gen95


def test_partial_generalization():
    rng = np.random.default_rng(2)
    train, test = synthetic_curves(STEPS, t_mem=300, t_star=2000, width=0.2, low=0.0, high=0.6, noise=0.01, rng=rng)
    d = detect(STEPS, train, test)
    assert d.outcome == "partial"
    assert d.t_grok is not None and d.censored_gen95


def test_no_generalization_is_censored():
    rng = np.random.default_rng(3)
    train, _ = synthetic_curves(STEPS, t_mem=300, t_star=1500, width=0.15, low=0.0, high=1.0, noise=0.0, rng=rng)
    test = np.clip(rng.normal(0.01, 0.005, len(STEPS)), 0, 1)
    d = detect(STEPS, train, test)
    assert d.outcome == "no_generalization" and d.censored and d.t_grok is None


def test_no_memorization():
    d = detect(STEPS, np.full(len(STEPS), 0.5), np.zeros(len(STEPS)))
    assert d.outcome == "no_memorization" and d.t_mem is None and not d.censored


def test_simultaneous():
    rng = np.random.default_rng(4)
    train, _ = synthetic_curves(STEPS, t_mem=300, t_star=1500, width=0.15, low=0.0, high=1.0, noise=0.0, rng=rng)
    d = detect(STEPS, train, train.copy())
    assert d.outcome == "simultaneous"


def test_stable_crossing_ignores_transient_hits():
    steps = np.arange(6)
    values = np.array([0.0, 0.97, 0.5, 0.96, 0.98, 0.99])
    assert stable_crossing(steps, values, 0.95) == 3
    assert stable_crossing(steps[:3], np.array([0.0, 0.97, 0.5]), 0.95) is None
