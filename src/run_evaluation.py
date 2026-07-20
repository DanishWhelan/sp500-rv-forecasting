"""
Evaluation entry point: run the model ladder through the walk-forward harness on the
real modelling data and write the comparison tables to results/.

This is the reproducible top-level script for the empirical comparison. Each model is
registered as a (name, factory, seeds) entry; the harness scores them identically and
saves forecasts, metrics (overall and per regime) and pairwise Diebold-Mariano tests.
New rungs of the ladder (GARCH, HAR-RV, LSTM, hybrid) are added to MODELS as built.

Usage:
    python src/run_evaluation.py
"""
import os

import pandas as pd

from harness import WalkForwardEvaluator
from ewma import EWMA

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "processed", "modelling_data.csv")

# (name, factory(seed) -> model, seeds). Deterministic models use a single seed.
MODELS = [
    ("ewma", lambda seed: EWMA(), (0,)),
]


def main():
    data = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} rows, {data.index.min().date()} to {data.index.max().date()}")

    evaluator = WalkForwardEvaluator(data)   # first forecast 2004-01-01, refit_every=1 (strict)
    for name, factory, seeds in MODELS:
        print(f"Running {name} (seeds={seeds}) ...")
        evaluator.run(factory, name=name, seeds=seeds)

    metrics = evaluator.save()
    print("\nSaved forecasts, metrics and DM tests to results/.\n")
    cols = ["model", "regime", "n", "qlike", "rmse"]
    print(metrics[cols].to_string(index=False,
          formatters={"qlike": "{:.5f}".format, "rmse": "{:.5f}".format}))


if __name__ == "__main__":
    main()
