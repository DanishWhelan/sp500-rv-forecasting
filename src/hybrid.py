"""
GARCH-LSTM hybrid (rung 6, the final rung).

The hybrid is the LSTM with one extra per-timestep input: the GARCH(1,1) one-step-ahead
conditional-variance forecast. It is NOT a new model class. model_base.VolatilityModel
makes feature ablation first-class, so the hybrid and its ablation arm differ only in the
`features` tuple they are constructed with:

    hybrid          features = (realised_variance, vix, garch_var)
    hybrid_nogarch  features = (realised_variance, vix)

This module builds the `garch_var` column and provides the two factories.

======================================================================================
THE LEAKAGE-CRITICAL POINT (this is the easiest leak in the whole thesis to introduce)
======================================================================================
Definition of the feature at row d:

    garch_var[d] = the GARCH(1,1) ONE-STEP-AHEAD FORECAST FOR day d,
                   estimated on data up to d-1.

It is NOT the in-sample fitted conditional variance from a GARCH fitted on the full
series. That would place a variance estimated WITH knowledge of day d (and every day
after it) into the predictor for day d, and would flatter the hybrid in exactly the way
CLAUDE.md forbids. arch_base deliberately exposes only forecast(), never the fitted
`conditional_volatility` path, so the leak is awkward to introduce by accident.

Alignment is inherited, not hand-rolled. walk_forward_forecast returns a frame INDEXED BY
TARGET DATE whose `forecast` column is the one-step forecast for that date built from
history strictly before it (harness.py, leak points 1-3). Joining that column on the index
therefore needs NO manual shifting: there is no off-by-one to get wrong, because the
harness already did the shift. Consequently the LSTM window ending at day t holds
garch_var[t-L+1 ... t], whose most recent element used data only through t-1, while the
target day is t+1. Every value in the window is knowable at day t.

Why precomputing the column once over the sample is legitimate:
    walk_forward_forecast at target i reads only data.iloc[:i], so row i's value is a
    function of the PREFIX alone. The column is therefore prefix-determined: building it
    on data[:k] yields values identical to building it on the full sample for every
    overlapping row. Rebuilding it from scratch inside each LSTM fit (the "purist"
    version) would return the same numbers at roughly 5000x the cost.
    This is asserted, not assumed: tests/test_hybrid.py::test_garch_feature_is_prefix_
    determined. If that test ever fails, precomputation is INVALID and this design must
    change.

Burn-in and the matched ablation arm:
    The column needs GARCH_BURNIN days before its first fit, so it starts partway into the
    sample and the hybrid loses early training rows. Because the hybrid-vs-ablation
    contrast IS the ablation result, the ablation arm (`hybrid_nogarch`) is run on the
    IDENTICAL row set rather than reusing the committed `lstm` run: otherwise a difference
    would confound the GARCH feature with training-sample size. Burn-in ends well before
    the 2004 test period, so the test sample and all cross-model DM comparisons are
    unaffected.
"""
import os

import pandas as pd

from garch import GARCH11
from harness import walk_forward_forecast
from lstm import LSTMModel

BASE = os.path.join(os.path.dirname(__file__), "..")
FEATURE_CACHE = os.path.join(BASE, "data", "processed", "garch_feature.csv")

GARCH_BURNIN = 250       # trading days of history before the first GARCH fit (~1 year)
GARCH_FEATURE = "garch_var"

HYBRID_FEATURES = ("realised_variance", "vix", GARCH_FEATURE)
NOGARCH_FEATURES = ("realised_variance", "vix")
# Both variance-scale columns enter the network in logs (see LSTMModel._raw_features).
LOG_FEATURES = ("realised_variance", GARCH_FEATURE)


def build_garch_feature(data, burnin=GARCH_BURNIN):
    """One-step-ahead GARCH(1,1) variance forecasts, indexed by the day they forecast.

    Runs the SAME leak-free walk-forward path used to score GARCH as a standalone model
    (strict re-fit at every origin, refit_every=1). The returned Series is indexed by
    target date, so `data.join(series)` aligns each day with the forecast made for it from
    data strictly before it.
    """
    if burnin < 1 or burnin >= len(data):
        raise ValueError(f"burnin={burnin} is not a valid offset into {len(data)} rows")

    # First forecast day is data.index[burnin], so the first fit sees rows 0..burnin-1.
    first = str(data.index[burnin].date())
    fc = walk_forward_forecast(GARCH11(), data, first_forecast_date=first, refit_every=1)
    return fc["forecast"].rename(GARCH_FEATURE)


def load_garch_feature(data, cache=FEATURE_CACHE, rebuild=False, burnin=GARCH_BURNIN):
    """Load the cached GARCH feature column, building (and caching) it if needed.

    Caching is safe because the column is prefix-determined (see the module docstring);
    the cache is a compute optimisation, not a change of definition. Pass cache=None to
    build without touching disk (used by the tests).
    """
    if cache is None:
        return build_garch_feature(data, burnin=burnin)
    if rebuild or not os.path.exists(cache):
        series = build_garch_feature(data, burnin=burnin)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        series.to_frame().to_csv(cache)
        return series
    return pd.read_csv(cache, index_col=0, parse_dates=True)[GARCH_FEATURE]


def hybrid_frame(data, cache=FEATURE_CACHE, rebuild=False, burnin=GARCH_BURNIN):
    """Modelling frame joined with the GARCH feature, burn-in rows dropped.

    BOTH hybrid arms run on this frame, so the ablation contrast isolates the GARCH
    feature and nothing else.
    """
    series = load_garch_feature(data, cache=cache, rebuild=rebuild, burnin=burnin)
    frame = data.join(series, how="left")
    frame = frame.dropna(subset=[GARCH_FEATURE])
    if frame.empty:
        raise ValueError("no rows left after joining the GARCH feature; check the cache")
    return frame


def make_hybrid(seed=0, with_garch=True, name=None, **kw):
    """Factory for the hybrid (with_garch=True) and its ablation arm (False).

    The two differ ONLY in the `features` tuple, per the model_base ablation contract.
    """
    features = HYBRID_FEATURES if with_garch else NOGARCH_FEATURES
    resolved = name or ("hybrid" if with_garch else "hybrid_nogarch")
    return LSTMModel(seed=seed, features=features, log_features=LOG_FEATURES,
                     name=resolved, **kw)
