"""Unit tests for the GARCH(1,1) baseline.

Key checks:
  - arch recovers known parameters from a simulated GARCH(1,1) series (sanity that the
    fit + scaling are wired up correctly);
  - the manual one-step recursion (used on the warm-parameter path and later by the
    hybrid) matches arch's own one-step forecast;
  - the forecast is a positive variance on the realised-variance scale.
"""
import numpy as np
import pandas as pd
import pytest

from garch import GARCH11, RETURN_SCALE
from harness import walk_forward_forecast


def _simulate_garch(n, omega, alpha, beta, seed=0):
    """Simulate a GARCH(1,1) return series (decimal returns)."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    s2 = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            s2 = omega + alpha * e[t - 1] ** 2 + beta * s2
        e[t] = np.sqrt(s2) * rng.standard_normal()
    return e


def _frame(returns):
    idx = pd.bdate_range("2000-01-03", periods=len(returns))
    return pd.DataFrame({"log_return": returns,
                         "realised_variance": returns ** 2 + 1e-8}, index=idx)


@pytest.fixture(scope="module")
def garch_sim_frame():
    # realistic daily params: unconditional vol ~ sqrt(1e-6/0.02) ~ 0.7% daily
    r = _simulate_garch(4000, omega=1e-6, alpha=0.08, beta=0.90, seed=7)
    return _frame(r)


def test_arch_recovers_known_parameters(garch_sim_frame):
    model = GARCH11()
    model.fit(garch_sim_frame)
    p = model.param_history[-1]
    # scale-free params should be recovered to within finite-sample tolerance
    assert p["alpha"] == pytest.approx(0.08, abs=0.04)
    assert p["beta"] == pytest.approx(0.90, abs=0.05)
    assert 0.0 < p["persistence"] < 1.0        # stationary
    assert p["converged"]


def test_manual_recursion_matches_arch(garch_sim_frame):
    model = GARCH11()
    model.fit(garch_sim_frame)
    # arch's own one-step forecast (percent^2)
    fc = model._res.forecast(horizon=1, reindex=False)
    arch_h = float(np.asarray(fc.variance)[-1, 0])
    manual_h = model._manual_one_step(model._scaled_returns(garch_sim_frame))
    assert manual_h == pytest.approx(arch_h, rel=1e-3)


def test_forecast_positive_and_on_target_scale(garch_sim_frame):
    model = GARCH11()
    model.fit(garch_sim_frame)
    h = model.forecast(garch_sim_frame)
    assert np.isfinite(h) and h > 0.0
    # decimal^2 daily variance: for ~0.7% daily vol, variance ~ 5e-5
    assert 1e-6 < h < 1e-3


def test_forecast_before_fit_raises(garch_sim_frame):
    with pytest.raises(RuntimeError):
        GARCH11().forecast(garch_sim_frame)


def test_garch_runs_through_harness_short():
    # a short expanding-origin run to confirm interface compliance (kept small for speed)
    r = _simulate_garch(900, omega=1e-6, alpha=0.08, beta=0.90, seed=3)
    data = _frame(r)
    start = data.index[700]
    fc = walk_forward_forecast(GARCH11(), data, first_forecast_date=str(start.date()))
    assert (fc["forecast"] > 0).all()
    assert not fc[["forecast", "actual"]].isna().any().any()
