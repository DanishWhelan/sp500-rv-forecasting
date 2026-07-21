"""
Self-tests for the LSTM model, with the leakage-sensitive points as the headline checks:
windowing boundary, forecast invariance to future data, scaler fit on train only, and
seed determinism. Kept small/fast (tiny net, few epochs) -- these test correctness and the
leak boundary, not predictive skill.
"""
import numpy as np
import pandas as pd
import pytest

from lstm import LSTMModel
from harness import walk_forward_forecast


def _data(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    rv = 1e-4 * np.exp(rng.normal(0, 0.4, n))
    vix = 15 + 5 * rng.random(n)
    return pd.DataFrame({"realised_variance": rv, "vix": vix}, index=idx)


def _fast_model(**kw):
    # small + short so tests run quickly; correctness/leak checks don't need skill
    params = dict(seed=0, lookback=10, hidden=8, max_epochs=5, patience=3)
    params.update(kw)
    return LSTMModel(**params)


# ======================================================================================
# LEAKAGE BOUNDARY
# ======================================================================================
def test_windowing_target_never_in_its_own_window():
    """On strictly increasing series, every input window's max is strictly below its target
    -- impossible if the window (ending at s) had included the target day s+1."""
    feats = np.arange(1, 200, dtype=float).reshape(-1, 1)     # strictly increasing
    target = np.arange(1, 200, dtype=float)
    X, y, ends = LSTMModel._make_sequences(feats, target, lookback=10)
    for i in range(len(X)):
        assert X[i].max() < y[i]                              # every predictor < its target
        assert y[i] == target[ends[i] + 1]                   # target is exactly one day ahead


def test_forecast_invariant_to_future_corruption():
    """The forecast for the day after the origin must not change when data at/after the
    target day is corrupted -- proof the model never reads beyond the history it is given."""
    data = _data()
    origin = 200
    model = _fast_model()
    model.fit(data.iloc[:origin])                            # train on history up to origin-1
    clean = model.forecast(data.iloc[:origin])

    corrupted = data.copy()
    corrupted.iloc[origin:] = 1.0                            # garbage from the target day onward
    after = model.forecast(corrupted.iloc[:origin])          # same history slice

    assert clean == after


def test_scaler_fitted_on_train_only():
    """The standardiser must reflect the training window passed to fit(), not the full series."""
    data = _data(n=400)
    k = 200
    model = _fast_model()
    model.fit(data.iloc[:k])

    train_logrv = np.log(data["realised_variance"].to_numpy()[:k])
    full_logrv = np.log(data["realised_variance"].to_numpy())
    assert model._scaler["tgt_mean"] == pytest.approx(train_logrv.mean())
    # and it is genuinely different from the full-series statistic (so full data was not used)
    assert abs(model._scaler["tgt_mean"] - full_logrv.mean()) > 1e-6


# ======================================================================================
# DETERMINISM (seed handling)
# ======================================================================================
def test_same_seed_identical_forecast():
    data = _data()
    a, b = _fast_model(seed=7), _fast_model(seed=7)
    a.fit(data.iloc[:200])
    b.fit(data.iloc[:200])
    assert a.forecast(data.iloc[:200]) == b.forecast(data.iloc[:200])


def test_different_seed_different_forecast():
    data = _data()
    a, b = _fast_model(seed=1), _fast_model(seed=2)
    a.fit(data.iloc[:200])
    b.fit(data.iloc[:200])
    assert a.forecast(data.iloc[:200]) != b.forecast(data.iloc[:200])


# ======================================================================================
# INTERFACE / BEHAVIOUR
# ======================================================================================
def test_forecast_positive_variance():
    data = _data()
    model = _fast_model()
    model.fit(data.iloc[:200])
    h = model.forecast(data.iloc[:200])
    assert np.isfinite(h) and h > 0.0


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_model().forecast(_data())


def test_vix_ablation_changes_feature_count():
    full = _fast_model(features=("realised_variance", "vix"))
    ablated = _fast_model(features=("realised_variance",))
    data = _data()
    full.fit(data.iloc[:200])
    ablated.fit(data.iloc[:200])
    assert full._scaler["feat_mean"].shape[0] == 2
    assert ablated._scaler["feat_mean"].shape[0] == 1


def test_runs_through_harness_warm_path():
    # refit once, then forecast many origins on the warm path (fixed net, growing history)
    data = _data(n=300)
    start = data.index[200]
    fc = walk_forward_forecast(_fast_model(), data,
                               first_forecast_date=str(start.date()), refit_every=10_000)
    assert (fc["forecast"] > 0).all()
    assert not fc[["forecast", "actual"]].isna().any().any()
