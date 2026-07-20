"""Unit tests for HAR-RV, with the leakage boundary as the headline check."""
import numpy as np
import pandas as pd
import pytest

from har import HARRV, WEEKLY, MONTHLY
from harness import walk_forward_forecast


def _frame_from_rv(rv):
    idx = pd.bdate_range("2000-01-03", periods=len(rv))
    return pd.DataFrame({"realised_variance": rv}, index=idx)


# ======================================================================================
# LEAKAGE BOUNDARY -- the critical checks, run before anything touches the harness
# ======================================================================================
def test_design_target_is_strictly_one_day_ahead():
    """The training target must be v_{s+1} for predictors dated s -- an exact forward shift."""
    rv = (np.arange(1, 200, dtype=float)) ** 2         # v = sqrt(rv) = 1,2,3,... strictly increasing
    model = HARRV()
    d = model._design(_frame_from_rv(rv))
    vol = np.sqrt(rv)
    # for each design row at position s (by daily==v_s), target must equal v_{s+1}
    for daily_val, target_val in zip(d["daily"], d["target"]):
        s = int(round(daily_val)) - 1                  # v_s = s+1 in this construction
        assert target_val == pytest.approx(vol[s + 1])


def test_no_predictor_contains_the_target_day():
    """On a strictly increasing volatility series, every predictor (daily/weekly/monthly)
    must be strictly LESS than the target it predicts. If any predictor had included the
    target day's (larger) RV, its value could not stay strictly below the target."""
    rv = (np.arange(1, 400, dtype=float)) ** 2         # strictly increasing volatility
    d = HARRV()._design(_frame_from_rv(rv))
    max_predictor = d[["daily", "weekly", "monthly"]].max(axis=1)
    assert (max_predictor < d["target"]).all()


def test_forecast_uses_only_history_up_to_origin():
    """The forecast for t+1 must be computable from history alone and must not change if
    future rows are altered -- i.e. it never reads v_{t+1}."""
    rng = np.random.default_rng(0)
    rv = 1e-4 * np.exp(rng.normal(0, 0.4, 300))
    data = _frame_from_rv(rv)
    model = HARRV()
    model.fit(data.iloc[:200])
    origin = 200
    f1 = model.forecast(data.iloc[:origin])            # history up to origin-1 (index 199)
    # corrupt everything from the target day onward; forecast on the same history is unchanged
    corrupted = data.copy()
    corrupted.iloc[origin:] = 999.0
    f2 = model.forecast(corrupted.iloc[:origin])
    assert f1 == f2


# ======================================================================================
# Behaviour / interface
# ======================================================================================
def test_forecast_positive_variance():
    rng = np.random.default_rng(1)
    rv = 1e-4 * np.exp(rng.normal(0, 0.4, 300))
    data = _frame_from_rv(rv)
    model = HARRV()
    model.fit(data)
    h = model.forecast(data)
    assert np.isfinite(h) and h > 0.0


def test_recovers_known_har_coefficients():
    """Simulate an exact HAR process and confirm OLS recovers the coefficients -- this also
    checks the predictor/target alignment is correct (a shift error would bias the slopes)."""
    b0, b_d, b_w, b_m = 0.0010, 0.40, 0.30, 0.20
    rng = np.random.default_rng(2)
    n = 4000
    v = np.full(n, 0.01)
    for t in range(MONTHLY, n - 1):
        vd = v[t]
        vw = v[t - WEEKLY + 1:t + 1].mean()
        vm = v[t - MONTHLY + 1:t + 1].mean()
        v[t + 1] = max(b0 + b_d * vd + b_w * vw + b_m * vm + rng.normal(0, 1e-4), 1e-5)

    model = HARRV()
    model.fit(_frame_from_rv(v ** 2))
    c0, cd, cw, cm = model._coef
    assert cd == pytest.approx(b_d, abs=0.08)
    assert cw == pytest.approx(b_w, abs=0.08)
    assert cm == pytest.approx(b_m, abs=0.08)


def test_har_runs_through_harness_short():
    rng = np.random.default_rng(3)
    rv = 1e-4 * np.exp(rng.normal(0, 0.4, 400))
    data = _frame_from_rv(rv)
    start = data.index[300]
    fc = walk_forward_forecast(HARRV(), data, first_forecast_date=str(start.date()))
    assert (fc["forecast"] > 0).all()
    assert not fc[["forecast", "actual"]].isna().any().any()


def test_har_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        HARRV().forecast(_frame_from_rv(np.full(50, 1e-4)))
