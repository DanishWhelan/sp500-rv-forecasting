"""Unit tests for the asymmetric models EGARCH(1,1) and GJR-GARCH(1,1)."""
import numpy as np
import pandas as pd
import pytest

from egarch import EGARCH11
from gjr import GJRGARCH11
from harness import walk_forward_forecast


def _simulate_gjr(n, omega, alpha, gamma, beta, seed=0):
    """Simulate a GJR-GARCH(1,1) return series with a leverage effect (gamma > 0)."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    s2 = omega / (1.0 - alpha - beta - 0.5 * gamma)
    for t in range(n):
        if t > 0:
            ind = 1.0 if e[t - 1] < 0 else 0.0
            s2 = omega + alpha * e[t - 1] ** 2 + gamma * e[t - 1] ** 2 * ind + beta * s2
        e[t] = np.sqrt(s2) * rng.standard_normal()
    return e


def _frame(returns):
    idx = pd.bdate_range("2000-01-03", periods=len(returns))
    return pd.DataFrame({"log_return": returns,
                         "realised_variance": returns ** 2 + 1e-8}, index=idx)


@pytest.fixture(scope="module")
def gjr_leverage_frame():
    r = _simulate_gjr(4000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.90, seed=11)
    return _frame(r)


def test_gjr_recovers_positive_leverage(gjr_leverage_frame):
    model = GJRGARCH11()
    model.fit(gjr_leverage_frame)
    p = model.param_history[-1]
    assert p["gamma"] > 0.0                       # leverage detected
    assert p["gamma"] == pytest.approx(0.10, abs=0.07)
    assert 0.0 < p["persistence"] < 1.0
    assert p["converged"]


def test_gjr_forecast_positive_on_scale(gjr_leverage_frame):
    model = GJRGARCH11()
    model.fit(gjr_leverage_frame)
    h = model.forecast(gjr_leverage_frame)
    assert np.isfinite(h) and 1e-6 < h < 1e-3


def test_egarch_fits_and_reports_gamma(gjr_leverage_frame):
    model = EGARCH11()
    model.fit(gjr_leverage_frame)
    p = model.param_history[-1]
    assert set(p) >= {"omega", "alpha", "gamma", "beta", "persistence", "converged"}
    assert np.isfinite(p["gamma"])
    # EGARCH persistence is the log-variance AR coefficient beta
    assert p["persistence"] == p["beta"]


def test_egarch_forecast_positive_on_scale(gjr_leverage_frame):
    model = EGARCH11()
    model.fit(gjr_leverage_frame)
    h = model.forecast(gjr_leverage_frame)
    assert np.isfinite(h) and 1e-6 < h < 1e-3


@pytest.mark.parametrize("factory", [EGARCH11, GJRGARCH11])
def test_asymmetric_runs_through_harness_short(factory):
    r = _simulate_gjr(900, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.90, seed=5)
    data = _frame(r)
    start = data.index[700]
    fc = walk_forward_forecast(factory(), data, first_forecast_date=str(start.date()))
    assert (fc["forecast"] > 0).all()
    assert not fc[["forecast", "actual"]].isna().any().any()
