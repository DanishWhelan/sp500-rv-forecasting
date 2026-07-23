"""
Time the five classical models through the walk-forward harness and record wall-clock to
results/timings.csv, so the runtimes cited in the write-up are reproducible rather than
transcribed from console output.

Times ONLY the deterministic classical ladder (EWMA, GARCH, EGARCH, GJR, HAR), each with
strict re-fit (refit_every=1) over all 4005 origins, matching how they are scored in
run_evaluation.py. The deep models are not re-timed here: the LSTM costs ~3.5h and the
hybrid arms were already measured this session (see the header note written into the CSV).

Timing is wall-clock of the full walk-forward (fit + forecast at every origin), captured by
the harness itself (WalkForwardEvaluator.timings), on the machine and env this is run on, so
absolute seconds are hardware-dependent; the ms/origin and cross-model ratios are the
portable quantities.

Usage: python src/time_classical.py
"""
import os
import platform
from datetime import date

import pandas as pd

from harness import WalkForwardEvaluator
from ewma import EWMA
from garch import GARCH11
from egarch import EGARCH11
from gjr import GJRGARCH11
from har import HARRV

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "processed", "modelling_data.csv")
RES = os.path.join(BASE, "results")

# (name, factory) for the deterministic classical ladder; all use strict re-fit (=1).
MODELS = [
    ("ewma", lambda seed: EWMA()),
    ("garch", lambda seed: GARCH11()),
    ("egarch", lambda seed: EGARCH11()),
    ("gjr", lambda seed: GJRGARCH11()),
    ("har", lambda seed: HARRV()),
]


def main():
    data = pd.read_csv(DATA, index_col=0, parse_dates=True)
    evaluator = WalkForwardEvaluator(data)   # first forecast 2004-01-01, refit_every=1

    rows = []
    run_date = date.today().isoformat()
    env = f"py{platform.python_version()}"
    for name, factory in MODELS:
        print(f"Timing {name} (refit_every=1, single seed) ...", flush=True)
        fc = evaluator.run(factory, name=name, seeds=(0,), refit_every=1)
        secs = evaluator.timings[name]
        n = len(fc)
        rows.append(dict(model=name, n_origins=n, n_seeds=1, refit_every=1,
                         wall_clock_s=round(secs, 2), ms_per_origin=round(1000 * secs / n, 3),
                         run_date=run_date, env=env))
        print(f"  {secs:.2f}s over {n} origins ({1000 * secs / n:.2f} ms/origin)", flush=True)

    out = pd.DataFrame(rows)
    path = os.path.join(RES, "timings.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
    print(out.to_string(index=False))
    print("\nDeep-model reference (measured this session, NOT re-timed here): "
          "hybrid 5-seed walk-forward = 13001.9s (~3.6h); hybrid_nogarch = 11785.3s; "
          "GARCH feature-column build = 65.2s.")


if __name__ == "__main__":
    main()
