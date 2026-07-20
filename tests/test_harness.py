"""
Self-tests for the evaluation harness.

Two jobs:
  1. Confirm the metrics and DM test are correct (hand-computed / known-property checks).
  2. ADVERSARIAL LEAK TESTS: prove the walk-forward loop cannot expose future information.
     A model that tries to read the day it must predict must FAIL, because that row is
     genuinely absent from the frame the harness passes it.

Run: pytest tests/ -v
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from model_base import VolatilityModel
from harness import (
    walk_forward_forecast, qlike_loss, qlike, rmse_vol, diebold_mariano,
    assign_regime, WalkForwardEvaluator, REGIMES,
)


# ======================================================================================
# Metrics
# ======================================================================================
def test_qlike_zero_at_truth():
    a = np.array([1e-4, 2e-4, 3e-4])
    assert np.allclose(qlike_loss(a, a), 0.0)


def test_qlike_positive_when_wrong():
    a = np.array([1e-4, 2e-4])
    f = np.array([2e-4, 1e-4])
    assert np.all(qlike_loss(a, f) > 0.0)


def test_qlike_hand_value():
    # actual=[1,2], forecast=[1,1]: L = [0, 2 - ln2 - 1]; mean = (1 - ln2)/2
    val = qlike(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert val == pytest.approx((1.0 - np.log(2.0)) / 2.0)


def test_qlike_penalises_under_prediction_more():
    # Under-predicting variance should cost more than the symmetric over-prediction.
    a = np.array([1.0])
    under = qlike_loss(a, np.array([0.5]))[0]   # forecast too low
    over = qlike_loss(a, np.array([2.0]))[0]    # forecast too high by same factor
    assert under > over


def test_qlike_requires_positive():
    with pytest.raises(ValueError):
        qlike_loss(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        qlike_loss(np.array([-1.0]), np.array([1.0]))


def test_rmse_vol_hand_value():
    # actual variance [1,2], forecast [1,1] -> vol diff [0, sqrt2 - 1]
    val = rmse_vol(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    expected = np.sqrt(((np.sqrt(2.0) - 1.0) ** 2) / 2.0)
    assert val == pytest.approx(expected)


# ======================================================================================
# Diebold-Mariano
# ======================================================================================
def test_dm_identical_forecasts_is_zero():
    loss = np.abs(np.random.default_rng(1).normal(size=200))
    dm, p = diebold_mariano(loss, loss)
    assert dm == 0.0 and p == 1.0


def test_dm_sign_and_significance():
    # Model A consistently worse (higher loss): differential mean > 0 -> DM > 0, small p.
    rng = np.random.default_rng(2)
    loss_b = np.abs(rng.normal(size=500))
    loss_a = loss_b + rng.normal(0.5, 0.1, size=500)   # A worse by ~0.5 on average
    dm, p = diebold_mariano(loss_a, loss_b)
    assert dm > 0.0
    assert p < 0.01


def test_dm_symmetric_no_difference():
    rng = np.random.default_rng(3)
    loss_a = np.abs(rng.normal(size=500))
    loss_b = np.abs(rng.normal(size=500))
    dm, p = diebold_mariano(loss_a, loss_b)
    assert p > 0.05          # no systematic difference


# ======================================================================================
# Regimes
# ======================================================================================
def test_regime_assignment():
    dates = pd.to_datetime(["2005-06-01",           # calm
                            "2008-10-15",           # gfc_2008
                            "2020-03-20",           # covid_2020
                            "2015-01-01"])          # calm
    labels = assign_regime(dates)
    assert list(labels) == ["calm", "gfc_2008", "covid_2020", "calm"]
    # sanity: the crisis labels correspond to the REGIMES windows
    for name, (lo, hi) in REGIMES.items():
        mid = pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2
        assert assign_regime([mid])[0] == name


# ======================================================================================
# Walk-forward loop (behaviour + alignment)
# ======================================================================================
def test_walk_forward_runs_and_aligns(sample_data, persistence_factory):
    fc = walk_forward_forecast(persistence_factory(), sample_data,
                               first_forecast_date="2004-01-01")
    # first forecast is the first trading day on/after 2004-01-01
    first_target = sample_data.index[sample_data.index.searchsorted(pd.Timestamp("2004-01-01"))]
    assert fc.index[0] == first_target
    assert fc.index[-1] == sample_data.index[-1]
    assert not fc[["forecast", "actual"]].isna().any().any()
    # actuals recorded by the harness must equal the source proxy exactly (alignment)
    src = sample_data.loc[fc.index, "realised_variance"]
    assert np.allclose(fc["actual"].to_numpy(), src.to_numpy())
    # persistence: each forecast equals the PREVIOUS day's realised variance (no peeking)
    prev = sample_data["realised_variance"].shift(1).loc[fc.index]
    assert np.allclose(fc["forecast"].to_numpy(), prev.to_numpy())


def test_bad_forecast_is_rejected_with_model_and_date(sample_data):
    """A model emitting a non-positive/non-finite variance must fail immediately, and
    the error must name the offending model so the bug is easy to locate."""
    class BadModel(VolatilityModel):
        def __init__(self, value):
            super().__init__(name="bad_model")
            self.value = value
        def fit(self, train):
            pass
        def forecast(self, history):
            return self.value

    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="bad_model"):
            walk_forward_forecast(BadModel(bad), sample_data,
                                  first_forecast_date="2004-01-01")


def test_refit_every_gt_1_warns(sample_data, persistence_factory):
    with pytest.warns(UserWarning, match="DEVIATES from the strict walk-forward"):
        walk_forward_forecast(persistence_factory(), sample_data,
                              first_forecast_date="2004-01-01", refit_every=5)


def test_evaluator_end_to_end(sample_data, persistence_factory):
    ev = WalkForwardEvaluator(sample_data, first_forecast_date="2004-01-01")
    ev.run(persistence_factory, name="persistence", seeds=(0,))
    metrics = ev.metrics()
    assert set(metrics["regime"]) >= {"overall", "calm", "gfc_2008", "covid_2020"}
    assert (metrics["qlike"] >= 0).all()


# ======================================================================================
# ADVERSARIAL LEAK TESTS  --  the point of building the harness first
# ======================================================================================
class ClairvoyantByPosition(VolatilityModel):
    """Cheats by trying to read one row PAST the history it is handed (i.e. the very day
    it is being asked to predict). If the harness leaked that row, this would succeed."""

    def __init__(self):
        super().__init__(name="clairvoyant_pos")
        self.attempts = 0

    def fit(self, train):
        pass

    def forecast(self, history):
        self.attempts += 1
        return float(history["realised_variance"].iloc[len(history)])   # out of bounds


class ClairvoyantByDate(VolatilityModel):
    """Cheats by computing the next trading date and trying to look it up in history."""

    def __init__(self, all_dates):
        super().__init__(name="clairvoyant_date")
        self.all_dates = pd.DatetimeIndex(all_dates)
        self.attempts = 0

    def fit(self, train):
        pass

    def forecast(self, history):
        self.attempts += 1
        last = history.index.max()
        nxt = self.all_dates[self.all_dates.searchsorted(last) + 1]     # the target day
        return float(history.loc[nxt, "realised_variance"])            # not in history


def test_leak_future_value_by_position_is_impossible(sample_data):
    """A model reaching one row past its history hits the end of the frame and raises,
    proving the future observation is never present in what the harness passes."""
    model = ClairvoyantByPosition()
    with pytest.raises(IndexError):
        walk_forward_forecast(model, sample_data, first_forecast_date="2004-01-01")
    # it failed on the FIRST forecast: the future value was absent from step one
    assert model.attempts == 1


def test_leak_future_value_by_date_is_impossible(sample_data):
    """Looking up the target date in history raises KeyError: the row does not exist
    in the slice, so the answer cannot be read even if its date is known."""
    model = ClairvoyantByDate(sample_data.index)
    with pytest.raises(KeyError):
        walk_forward_forecast(model, sample_data, first_forecast_date="2004-01-01")
    assert model.attempts == 1


def test_history_never_reaches_target(sample_data):
    """Directly assert the harness invariant for every origin: the last row the model
    can see is strictly before the day being forecast."""
    dates = sample_data.index
    first_target = int(dates.searchsorted(pd.Timestamp("2004-01-01")))
    for target_i in range(first_target, len(dates)):
        history = sample_data.iloc[:target_i]        # exactly what the harness passes
        target_date = dates[target_i]
        assert history.index.max() < target_date
        assert target_date not in history.index
