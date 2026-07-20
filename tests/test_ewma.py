"""Unit tests for the EWMA (RiskMetrics) baseline."""
import numpy as np
import pandas as pd
import pytest

from ewma import EWMA, RISKMETRICS_LAMBDA
from harness import walk_forward_forecast


def _frame(returns):
    idx = pd.bdate_range("2000-01-03", periods=len(returns))
    return pd.DataFrame({"log_return": returns, "realised_variance": np.abs(returns) + 1e-6},
                        index=idx)


def test_ewma_matches_manual_recursion():
    # r^2 = [0.01, 0.04, 0.0025]; seed = 0.01
    # step1: 0.94*0.01   + 0.06*0.04   = 0.0118
    # step2: 0.94*0.0118 + 0.06*0.0025 = 0.011242
    model = EWMA(lam=0.94)
    model.fit(_frame([0.1, -0.2, 0.05]))
    fc = model.forecast(_frame([0.1, -0.2, 0.05]))
    assert fc == pytest.approx(0.011242, rel=1e-9)


def test_ewma_constant_returns_give_constant_variance():
    c = 0.013
    model = EWMA()
    fc = model.forecast(_frame([c] * 500))
    assert fc == pytest.approx(c ** 2, rel=1e-9)


def test_ewma_default_lambda_is_riskmetrics():
    assert EWMA().lam == RISKMETRICS_LAMBDA == 0.94


def test_ewma_rejects_bad_lambda():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            EWMA(lam=bad)


def test_ewma_forecast_strictly_positive():
    model = EWMA()
    fc = model.forecast(_frame([0.0] * 300))   # degenerate: all-zero returns
    assert fc > 0.0                            # floored, never zero (keeps QLIKE defined)


def test_ewma_ignores_leading_nan_return():
    # A leading NaN (as produced by a diff before alignment) must not poison the recursion.
    with_nan = _frame([np.nan, 0.1, -0.2, 0.05])
    without = _frame([0.1, -0.2, 0.05])
    assert EWMA().forecast(with_nan) == pytest.approx(EWMA().forecast(without), rel=1e-12)


def test_ewma_runs_through_harness(sample_data):
    fc = walk_forward_forecast(EWMA(), sample_data, first_forecast_date="2004-01-01")
    assert not fc[["forecast", "actual"]].isna().any().any()
    assert (fc["forecast"] > 0).all()
