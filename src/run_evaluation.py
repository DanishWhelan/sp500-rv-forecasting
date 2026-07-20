"""
Evaluation entry point: run the model ladder through the walk-forward harness on the
real modelling data and write the comparison tables to results/.

Reproducible top-level script for the empirical comparison. Each model is registered as
a (name, factory, seeds) entry; the harness scores them identically and saves forecasts,
metrics (overall and per regime) and pairwise Diebold-Mariano tests. It also reports the
walk-forward wall-clock time per model (relevant to the LSTM tractability question) and,
for any model exposing a `param_history`, saves and summarises the fitted-parameter path
(the supervisor-required GARCH baseline validation).

Usage:
    python src/run_evaluation.py
"""
import os

import numpy as np
import pandas as pd

from harness import WalkForwardEvaluator
from ewma import EWMA
from garch import GARCH11
from egarch import EGARCH11
from gjr import GJRGARCH11

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "processed", "modelling_data.csv")
RES = os.path.join(BASE, "results")

# (name, factory(seed) -> model, seeds). Deterministic models use a single seed.
MODELS = [
    ("ewma", lambda seed: EWMA(), (0,)),
    ("garch", lambda seed: GARCH11(), (0,)),
    ("egarch", lambda seed: EGARCH11(), (0,)),
    ("gjr", lambda seed: GJRGARCH11(), (0,)),
]


def _summarise_params(name, model):
    """Save the full fitted-parameter path and print a summary (last fit + distribution
    across all refits) for a model that records `param_history`."""
    ph = pd.DataFrame(model.param_history)
    ph.to_csv(os.path.join(RES, f"{name}_params.csv"), index=False)

    last = ph.iloc[-1]
    rows = []
    # include the asymmetry term gamma when the model has one (EGARCH, GJR)
    param_cols = [c for c in ["omega", "alpha", "gamma", "beta", "persistence"]
                  if c in ph.columns]
    for col in param_cols:
        rows.append(dict(parameter=col, last_fit=last[col], mean=ph[col].mean(),
                         std=ph[col].std(), min=ph[col].min(), max=ph[col].max()))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RES, f"{name}_param_validation.csv"), index=False)

    n_fits = len(ph)
    n_bad = int((~ph["converged"]).sum())
    print(f"\n=== {name.upper()} fitted parameters over {n_fits} refits "
          f"({n_bad} non-converged) ===")
    print(summary.to_string(index=False, formatters={
        c: "{:.6f}".format for c in ["last_fit", "mean", "std", "min", "max"]}))
    return summary


def main():
    data = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} rows, {data.index.min().date()} to {data.index.max().date()}")

    evaluator = WalkForwardEvaluator(data)   # first forecast 2004-01-01, refit_every=1 (strict)
    for name, factory, seeds in MODELS:
        print(f"Running {name} (seeds={seeds}) ...")
        evaluator.run(factory, name=name, seeds=seeds)
        print(f"  walk-forward wall-clock: {evaluator.timings[name]:.1f}s")

    metrics = evaluator.save()

    print("\n=== Metrics (QLIKE primary, RMSE in volatility units) ===")
    cols = ["model", "regime", "n", "qlike", "rmse"]
    print(metrics[cols].to_string(index=False, formatters={
        "qlike": "{:.5f}".format, "rmse": "{:.5f}".format}))

    dm = evaluator.dm_tests()
    if len(dm):
        print("\n=== Diebold-Mariano (dm_stat>0 => model_a worse; p two-sided) ===")
        print(dm.to_string(index=False, formatters={
            "dm_stat": "{:.3f}".format, "p_value": "{:.4f}".format}))

    for name, _, _ in MODELS:
        model = evaluator.models.get(name)
        if model is not None and getattr(model, "param_history", None):
            _summarise_params(name, model)

    print("\n=== Walk-forward timing ===")
    for name in evaluator.timings:
        secs = evaluator.timings[name]
        n = int(metrics.loc[metrics["model"] == name, "n"].iloc[0])
        print(f"  {name:6s} {secs:7.1f}s over {n} origins "
              f"({1000 * secs / n:.1f} ms/origin)")


if __name__ == "__main__":
    main()
