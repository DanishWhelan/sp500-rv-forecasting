"""
Self-tests for the GARCH-LSTM hybrid, with the leak boundary as the headline.

The critical test is test_garch_feature_is_prefix_determined: it is what licenses building
the GARCH feature column ONCE over the sample instead of rebuilding it inside every LSTM
fit. If it fails, precomputation is invalid and src/hybrid.py must change.

Kept small/fast (short series, tiny net, few epochs): these check correctness and the leak
boundary, not predictive skill.
"""
import numpy as np
import pandas as pd
import pytest

from arch import arch_model

from garch import GARCH11, RETURN_SCALE
from harness import walk_forward_forecast
from hybrid import (GARCH_FEATURE, build_garch_feature, hybrid_frame, make_hybrid,
                    HYBRID_FEATURES, NOGARCH_FEATURES)
from lstm import LSTMModel


def _data(n=400, seed=0):
    """Synthetic frame with the columns the hybrid needs (returns drive GARCH)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    ret = rng.normal(0, 0.01, n)
    rv = 1e-4 * np.exp(rng.normal(0, 0.4, n))
    vix = 15 + 5 * rng.random(n)
    return pd.DataFrame(
        {"log_return": ret, "realised_variance": rv, "vix": vix}, index=idx)


def _fast_hybrid(**kw):
    params = dict(lookback=10, hidden=8, max_epochs=5, patience=3)
    params.update(kw)
    return make_hybrid(seed=0, **params)


# ======================================================================================
# LEAKAGE BOUNDARY
# ======================================================================================
def test_garch_feature_is_prefix_determined():
    """THE test that licenses precomputing the feature column.

    Building the column on a truncated prefix must reproduce, exactly, the values from
    building it on the full sample. This holds because walk_forward_forecast at target i
    reads only data.iloc[:i], so no row's value can depend on anything after it. If this
    fails, the cached column carries future information and the design is invalid.
    """
    data = _data(n=340)
    burnin = 250

    full = build_garch_feature(data, burnin=burnin)
    prefix = build_garch_feature(data.iloc[:300], burnin=burnin)

    common = full.index.intersection(prefix.index)
    assert len(common) > 20, "prefix test needs a meaningful overlap"
    np.testing.assert_allclose(full.loc[common].to_numpy(),
                               prefix.loc[common].to_numpy(), rtol=1e-9, atol=0.0)


def test_garch_feature_is_not_the_in_sample_fit():
    """The feature must be the one-step-ahead FORECAST, not the full-sample fitted
    conditional variance. The forbidden quantity is fitted with knowledge of the whole
    series, so it must differ materially from the walk-forward column."""
    data = _data(n=340)
    feature = build_garch_feature(data, burnin=250)

    # The forbidden path, constructed here ONLY to prove we are not producing it.
    r = data["log_return"].to_numpy(dtype=float) * RETURN_SCALE
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                     dist="normal", rescale=False).fit(disp="off", update_freq=0)
    in_sample = pd.Series(
        np.asarray(res.conditional_volatility) ** 2 / (RETURN_SCALE ** 2),
        index=data.index).loc[feature.index]

    # Not equal anywhere near numerical tolerance, and not merely a shifted copy.
    assert not np.allclose(feature.to_numpy(), in_sample.to_numpy(), rtol=1e-6)
    rel = np.abs(feature.to_numpy() - in_sample.to_numpy()) / in_sample.to_numpy()
    assert np.median(rel) > 1e-4


def test_feature_value_uses_only_strictly_earlier_data():
    """garch_var[d] must be unchanged when data from day d onward is corrupted: it is the
    forecast FOR day d, made from data strictly before d."""
    data = _data(n=320)
    burnin = 250
    clean = build_garch_feature(data, burnin=burnin)

    corrupt_from = 300
    corrupted = data.copy()
    corrupted.iloc[corrupt_from:] = 0.5           # garbage from day 300 onward
    after = build_garch_feature(corrupted, burnin=burnin)

    # Values for days at or before the corruption point are untouched.
    unaffected = clean.index[:corrupt_from - burnin]
    np.testing.assert_allclose(clean.loc[unaffected].to_numpy(),
                               after.loc[unaffected].to_numpy(), rtol=1e-9)


def test_hybrid_forecast_invariant_to_future_corruption():
    """End-to-end: the hybrid's forecast for the day after the origin must not move when
    data at/after the target day is corrupted (mirrors test_lstm.py)."""
    frame = hybrid_frame(_data(n=400), cache=None)
    origin = 100
    model = _fast_hybrid()
    model.fit(frame.iloc[:origin])
    clean = model.forecast(frame.iloc[:origin])

    corrupted = frame.copy()
    corrupted.iloc[origin:] = 1.0
    after = model.forecast(corrupted.iloc[:origin])

    assert clean == after


def test_hybrid_window_target_alignment_three_features():
    """No input window contains its own target day, with the 3-feature frame."""
    feats = np.column_stack([np.arange(1, 200, dtype=float)] * 3)
    target = np.arange(1, 200, dtype=float)
    X, y, ends = LSTMModel._make_sequences(feats, target, lookback=10)
    for i in range(len(X)):
        assert X[i].max() < y[i]
        assert y[i] == target[ends[i] + 1]


# ======================================================================================
# FEATURE CONSTRUCTION / ABLATION CONTRACT
# ======================================================================================
def test_garch_feature_enters_in_logs():
    """The GARCH feature is a variance and must be log-transformed like realised variance,
    not fed in raw."""
    frame = hybrid_frame(_data(n=340), cache=None)
    model = _fast_hybrid()
    raw = model._raw_features(frame)
    expected = np.log(frame[GARCH_FEATURE].to_numpy(dtype=float))
    np.testing.assert_allclose(raw[:, HYBRID_FEATURES.index(GARCH_FEATURE)], expected)


def test_ablation_differs_only_in_feature_count():
    frame = hybrid_frame(_data(n=340), cache=None)
    full, ablated = _fast_hybrid(with_garch=True), _fast_hybrid(with_garch=False)
    assert full.features == HYBRID_FEATURES
    assert ablated.features == NOGARCH_FEATURES
    full.fit(frame.iloc[:200])
    ablated.fit(frame.iloc[:200])
    assert full._scaler["feat_mean"].shape[0] == 3
    assert ablated._scaler["feat_mean"].shape[0] == 2


def test_plain_lstm_behaviour_unchanged_by_log_features_param():
    """The log_features default must reproduce the original behaviour exactly, so the
    already-committed LSTM results remain valid."""
    data = _data(n=300)
    model = LSTMModel(seed=0, features=("realised_variance", "vix"))
    raw = model._raw_features(data)
    np.testing.assert_allclose(raw[:, 0], np.log(data["realised_variance"].to_numpy()))
    np.testing.assert_allclose(raw[:, 1], data["vix"].to_numpy())   # VIX NOT logged


def test_hybrid_frame_drops_burnin_rows_only():
    data = _data(n=400)
    frame = hybrid_frame(data, cache=None)
    assert GARCH_FEATURE in frame.columns
    assert not frame[GARCH_FEATURE].isna().any()
    assert (frame[GARCH_FEATURE] > 0).all()
    assert frame.index[0] == data.index[250]        # burn-in dropped, nothing else
    assert frame.index[-1] == data.index[-1]


# ======================================================================================
# INTERFACE
# ======================================================================================
def test_hybrid_runs_through_harness():
    frame = hybrid_frame(_data(n=400), cache=None)   # 400 rows less 250 burn-in = 150
    start = frame.index[100]
    fc = walk_forward_forecast(_fast_hybrid(), frame,
                               first_forecast_date=str(start.date()), refit_every=10_000)
    assert (fc["forecast"] > 0).all()
    assert not fc[["forecast", "actual"]].isna().any().any()


def test_same_seed_identical_forecast():
    frame = hybrid_frame(_data(n=340), cache=None)
    a, b = _fast_hybrid(), _fast_hybrid()
    a.fit(frame.iloc[:200])
    b.fit(frame.iloc[:200])
    assert a.forecast(frame.iloc[:200]) == b.forecast(frame.iloc[:200])
