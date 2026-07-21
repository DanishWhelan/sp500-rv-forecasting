"""
Leakage-free walk-forward evaluation harness.

Built BEFORE any model, per CLAUDE.md. Provides:
  - walk_forward_forecast(): the single leak-critical expanding-origin loop, reused
    everywhere (including inside models that need leak-free derived features).
  - qlike_loss() / rmse_vol(): the scoring rules (QLIKE primary, RMSE for comparability).
  - diebold_mariano(): pairwise significance testing with the HLN small-sample correction.
  - assign_regime(): labels each date calm / gfc_2008 / covid_2020 from the canonical
    REGIMES constant in build_target.py.
  - WalkForwardEvaluator: runs one or more models across seeds, computes overall and
    per-regime metrics and pairwise DM tests, and writes everything to results/ as CSV.

Everything operates in VARIANCE space: models forecast a conditional variance, and the
target proxy is the Oxford-Man realised variance. QLIKE is computed on variance; RMSE is
computed on volatility (sqrt of both) for interpretability.

Leakage is prevented BY CONSTRUCTION, not by convention. See walk_forward_forecast().
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# REGIMES is the single source of truth for the crisis windows (defined in build_target).
try:
    from build_target import REGIMES
except ImportError:  # allow `import src.harness` as well as `import harness`
    from src.build_target import REGIMES

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")

PROXY_COL = "realised_variance"        # the realised-variance target the models predict
FIRST_FORECAST_DATE = "2004-01-01"     # ~4 years initial training; test spans 2004 onward
REGIME_ORDER = ["calm", *REGIMES.keys()]


# --------------------------------------------------------------------------------------
# Regime assignment
# --------------------------------------------------------------------------------------
def assign_regime(dates) -> np.ndarray:
    """Label each date 'calm' unless it falls inside a REGIMES crisis window."""
    idx = pd.DatetimeIndex(dates)
    labels = np.array(["calm"] * len(idx), dtype=object)
    for name, (lo, hi) in REGIMES.items():
        mask = np.asarray((idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi)))
        labels[mask] = name
    return labels


# --------------------------------------------------------------------------------------
# The leak-critical loop
# --------------------------------------------------------------------------------------
def walk_forward_forecast(model, data, first_forecast_date=FIRST_FORECAST_DATE,
                          refit_every=1, proxy_col=PROXY_COL):
    """Expanding-origin walk-forward. Returns a frame indexed by forecast date with
    columns [forecast, actual, regime].

    `history` is the FULL expanding window from the start of the sample (data.iloc[:t+1]),
    never a truncated/rolling slice; models that seed a recursion may rely on this. The
    model's forecast is validated to be a strictly positive, finite variance each step.

    Leakage is blocked at six structural points (see inline comments):
      1. At origin t the model is handed data.iloc[:t+1] and nothing else.
      2. The actual value at t+1 is held here and read only AFTER forecast() returns.
      3. An assertion enforces history.index.max() < target_date every step.
      4. No global scaling exists; each model scales inside fit() on its train slice.
      5. refit_every defaults to 1 (re-fit at every origin), per CLAUDE.md.
      6. Reused verbatim inside feature-building models, so derived features (e.g. the
         hybrid's GARCH feature) are one-step-ahead forecasts, never in-sample values.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    if refit_every > 1:
        warnings.warn(
            f"refit_every={refit_every} (>1): model parameters are NOT re-estimated at "
            f"every origin. This DEVIATES from the strict walk-forward protocol in "
            f"CLAUDE.md and is only justified for expensive models (e.g. the LSTM) on "
            f"tractability grounds. Document this compromise in the write-up.",
            stacklevel=2,
        )

    data = data.sort_index()
    dates = data.index
    if proxy_col not in data.columns:
        raise KeyError(f"proxy column {proxy_col!r} not in data")

    # First day to be predicted: the first date on/after first_forecast_date.
    first_target = int(dates.searchsorted(pd.Timestamp(first_forecast_date), side="left"))
    if first_target < 1:
        raise ValueError("first_forecast_date is at or before the start of the sample")
    if first_target >= len(dates):
        raise ValueError("first_forecast_date is after the end of the sample")

    records = []
    for step, target_i in enumerate(range(first_target, len(dates))):
        origin_i = target_i - 1
        history = data.iloc[:origin_i + 1]          # (1) rows 0..origin only
        target_date = dates[target_i]

        # (3) structural guard: the model can see nothing at or beyond the target day.
        assert history.index.max() == dates[origin_i]
        assert target_date > history.index.max()

        if step % refit_every == 0:                 # (5) strict re-fit by default
            model.fit(history)                      # (4) any scaling fit here, on train only

        forecast = float(model.forecast(history))   # variance forecast for target_i
        # Contract guard: models must emit a strictly positive, finite VARIANCE. Checked
        # here so a bad forecast fails immediately, naming the model and date, rather than
        # surfacing later as an opaque error inside the QLIKE computation.
        if not np.isfinite(forecast) or forecast <= 0.0:
            raise ValueError(
                f"{getattr(model, 'name', type(model).__name__)}: forecast for "
                f"{pd.Timestamp(target_date).date()} is {forecast!r}; models must return "
                f"a strictly positive, finite variance.")
        # (2) the answer is read ONLY now, purely for scoring; never passed to the model.
        actual = float(data[proxy_col].iloc[target_i])
        records.append((target_date, forecast, actual))

    out = pd.DataFrame.from_records(records, columns=["date", "forecast", "actual"])
    out = out.set_index("date")
    out["regime"] = assign_regime(out.index)
    return out


# --------------------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------------------
def qlike_loss(actual, forecast):
    """Per-observation QLIKE (Patton 2011, robust form), computed in VARIANCE space:

        L = actual/forecast - log(actual/forecast) - 1

    Non-negative, zero iff forecast == actual, and robust to noise in the realised
    variance proxy. Penalises under-prediction of variance more than over-prediction.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if np.any(f <= 0):
        raise ValueError("QLIKE requires strictly positive variance forecasts")
    if np.any(a <= 0):
        raise ValueError("QLIKE requires strictly positive realised variance")
    ratio = a / f
    return ratio - np.log(ratio) - 1.0


def qlike(actual, forecast) -> float:
    """Mean QLIKE over the sample."""
    return float(np.mean(qlike_loss(actual, forecast)))


def rmse_vol(actual, forecast) -> float:
    """RMSE in VOLATILITY space: sqrt-transform variance to volatility, then RMSE.

    More interpretable than variance-space RMSE and less dominated by crisis spikes.
    """
    a = np.sqrt(np.asarray(actual, dtype=float))
    f = np.sqrt(np.asarray(forecast, dtype=float))
    return float(np.sqrt(np.mean((f - a) ** 2)))


# --------------------------------------------------------------------------------------
# Diebold-Mariano test
# --------------------------------------------------------------------------------------
def diebold_mariano(loss_a, loss_b, h=1):
    """Diebold-Mariano test on two per-observation loss series (same loss, e.g. QLIKE).

    Tests H0: E[loss_a - loss_b] = 0. Returns (dm_stat, p_value), two-sided.

    Sign convention: dm_stat > 0 means model A has the HIGHER (worse) loss.
    Uses a Newey-West long-run variance with lag h-1 (0 for one-step forecasts) and the
    Harvey-Leybourne-Newbold small-sample correction, with p-values from t(T-1).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = d.size
    if T < 2:
        return np.nan, np.nan
    if np.allclose(d, 0.0):
        return 0.0, 1.0                              # identical forecasts: no difference

    dbar = d.mean()
    # Newey-West long-run variance (rectangular kernel up to lag h-1).
    lrv = np.mean((d - dbar) ** 2)
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        lrv += 2.0 * cov
    if lrv <= 0:
        return np.nan, np.nan

    dm = dbar / np.sqrt(lrv / T)
    # Harvey-Leybourne-Newbold small-sample correction.
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= hln
    p = 2.0 * stats.t.sf(np.abs(dm), df=T - 1)
    return float(dm), float(p)


# --------------------------------------------------------------------------------------
# Orchestration: run models, aggregate seeds, score, save
# --------------------------------------------------------------------------------------
def _regime_subsets(frame):
    """Yield (regime_label, sub_frame) for 'overall' plus each populated regime."""
    yield "overall", frame
    for regime in REGIME_ORDER:
        sub = frame[frame["regime"] == regime]
        if len(sub):
            yield regime, sub


class WalkForwardEvaluator:
    """Runs models through the walk-forward harness and produces the comparison tables.

    A model is supplied as a factory `make(seed) -> VolatilityModel` so that stochastic
    models (LSTM, hybrid) can be run across multiple seeds. Deterministic models ignore
    the seed and should be run with a single seed.
    """

    def __init__(self, data, first_forecast_date=FIRST_FORECAST_DATE, refit_every=1,
                 proxy_col=PROXY_COL, results_dir=RESULTS):
        self.data = data.sort_index()
        self.first_forecast_date = first_forecast_date
        self.refit_every = refit_every
        self.proxy_col = proxy_col
        self.results_dir = results_dir
        self._per_seed = {}     # name -> list of per-seed forecast frames
        self._forecast = {}     # name -> seed-averaged forecast frame (variance space)
        self.models = {}        # name -> last fitted model instance (for parameter inspection)
        self.timings = {}       # name -> wall-clock seconds for the walk-forward run

    def run(self, model_factory, name=None, seeds=(0,), refit_every=None):
        """Run one model across `seeds`; store per-seed and seed-averaged forecasts.

        `refit_every` overrides the evaluator default for this model only, so an expensive
        stochastic model (the LSTM) can re-estimate less often (e.g. quarterly) while the
        classical models stay strict (=1) in the same evaluation.
        """
        refit = self.refit_every if refit_every is None else refit_every
        frames = []
        resolved_name = name
        model = None
        t0 = time.perf_counter()
        for seed in seeds:
            model = model_factory(seed)
            resolved_name = name or model.name
            frames.append(walk_forward_forecast(
                model, self.data, self.first_forecast_date, refit, self.proxy_col))
        self.timings[resolved_name] = time.perf_counter() - t0
        self.models[resolved_name] = model    # last seed's instance (holds fitted state)

        agg = frames[0].copy()
        if len(frames) > 1:
            stacked = np.column_stack([f["forecast"].to_numpy() for f in frames])
            agg["forecast"] = stacked.mean(axis=1)   # average variance forecast across seeds
        self._per_seed[resolved_name] = frames
        self._forecast[resolved_name] = agg
        return agg

    def metrics(self) -> pd.DataFrame:
        """Overall and per-regime QLIKE and RMSE for every model.

        For multi-seed models, reports mean and std across seeds (std is 0 for a single
        seed / deterministic model).
        """
        rows = []
        for name, frames in self._per_seed.items():
            per = {}
            for fr in frames:
                for regime, sub in _regime_subsets(fr):
                    per.setdefault(regime, []).append(
                        (qlike(sub["actual"], sub["forecast"]),
                         rmse_vol(sub["actual"], sub["forecast"]),
                         len(sub)))
            for regime, vals in per.items():
                qs = [v[0] for v in vals]
                rs = [v[1] for v in vals]
                rows.append(dict(
                    model=name, regime=regime, n=vals[0][2], n_seeds=len(frames),
                    qlike=float(np.mean(qs)), qlike_std=float(np.std(qs)),
                    rmse=float(np.mean(rs)), rmse_std=float(np.std(rs))))
        return pd.DataFrame(rows)

    def dm_tests(self) -> pd.DataFrame:
        """Pairwise Diebold-Mariano on seed-averaged forecasts, overall and per regime."""
        rows = []
        names = list(self._forecast)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                fa, fb = self._forecast[a], self._forecast[b]
                common = fa.index.intersection(fb.index)
                regimes = assign_regime(common)
                for regime in ["overall", *REGIME_ORDER]:
                    idx = common if regime == "overall" else common[regimes == regime]
                    if len(idx) < 2:
                        continue
                    la = qlike_loss(fa.loc[idx, "actual"], fa.loc[idx, "forecast"])
                    lb = qlike_loss(fb.loc[idx, "actual"], fb.loc[idx, "forecast"])
                    dm, p = diebold_mariano(la, lb)
                    rows.append(dict(model_a=a, model_b=b, regime=regime, n=len(idx),
                                     dm_stat=dm, p_value=p))
        return pd.DataFrame(rows)

    def save(self):
        """Write forecasts, metrics and DM tests to results/ as CSV."""
        fc_dir = os.path.join(self.results_dir, "forecasts")
        os.makedirs(fc_dir, exist_ok=True)
        for name, frame in self._forecast.items():
            frame.to_csv(os.path.join(fc_dir, f"{name}.csv"))
        metrics = self.metrics()
        metrics.to_csv(os.path.join(self.results_dir, "metrics_by_regime.csv"), index=False)
        metrics[metrics["regime"] == "overall"].to_csv(
            os.path.join(self.results_dir, "metrics_overall.csv"), index=False)
        self.dm_tests().to_csv(os.path.join(self.results_dir, "dm_tests.csv"), index=False)
        return metrics
