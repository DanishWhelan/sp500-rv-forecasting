"""
Run the final rung: the GARCH-LSTM hybrid and its matched GARCH-feature ablation arm.

Two arms, identical in every respect except the GARCH feature (see src/hybrid.py for the
leak argument and the burn-in / matched-arm reasoning):

    hybrid          features = (realised_variance, vix, garch_var)
    hybrid_nogarch  features = (realised_variance, vix)

Both run on the SAME frame (burn-in rows dropped), so the contrast isolates the GARCH
feature and not training-sample size. This is why the already-committed `lstm` run is not
reused as the ablation arm: it trains on ~250 extra early rows.

Protocol matches the LSTM: 5 seeds, monthly refit (21). The GARCH feature column itself is
built with STRICT re-fit (refit_every=1) and cached.

Canonical tables are updated rather than rebuilt from scratch: the two new models' metric
rows are merged into results/metrics_by_regime.csv, and the pairwise Diebold-Mariano table
is recomputed across ALL saved forecasts so every pair is present and consistent. No
existing model is re-run.

Usage: python src/run_hybrid.py [--rebuild-feature]
"""
import argparse
import os

import pandas as pd

from harness import (WalkForwardEvaluator, qlike_loss, diebold_mariano, assign_regime,
                     REGIME_ORDER)
from hybrid import hybrid_frame, make_hybrid

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "processed", "modelling_data.csv")
RES = os.path.join(BASE, "results")
FC = os.path.join(RES, "forecasts")

SEEDS = (0, 1, 2, 3, 4)
REFIT_EVERY = 21
ARMS = [("hybrid", True), ("hybrid_nogarch", False)]


def _merge_metrics(new_rows):
    """Replace the rows for the newly-run models in metrics_by_regime.csv, keep the rest."""
    path = os.path.join(RES, "metrics_by_regime.csv")
    new_names = set(new_rows["model"])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[~existing["model"].isin(new_names)]
        merged = pd.concat([existing, new_rows], ignore_index=True)
    else:
        merged = new_rows
    merged.to_csv(path, index=False)
    merged[merged["regime"] == "overall"].to_csv(
        os.path.join(RES, "metrics_overall.csv"), index=False)
    return merged


def _recompute_dm_tests():
    """Full pairwise DM across every saved forecast file (seed-averaged), per regime.

    Recomputed from disk so the table covers all pairs, including the new hybrid arms
    against models that were run in earlier sessions.
    """
    names = sorted(f[:-4] for f in os.listdir(FC) if f.endswith(".csv"))
    frames = {n: pd.read_csv(os.path.join(FC, f"{n}.csv"), index_col=0, parse_dates=True)
              for n in names}

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = frames[a], frames[b]
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
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "dm_tests.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-feature", action="store_true",
                    help="rebuild the cached GARCH feature column from scratch")
    args = ap.parse_args()

    data = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} rows, {data.index.min().date()} to {data.index.max().date()}")

    print("Building/loading the one-step GARCH feature column (strict refit) ...", flush=True)
    frame = hybrid_frame(data, rebuild=args.rebuild_feature)
    print(f"  hybrid frame: {len(frame)} rows, {frame.index.min().date()} to "
          f"{frame.index.max().date()} ({len(data) - len(frame)} burn-in rows dropped)")

    evaluator = WalkForwardEvaluator(data)
    forecasts = {}
    for name, with_garch in ARMS:
        print(f"Running {name} (seeds={SEEDS}, refit_every={REFIT_EVERY}) ...", flush=True)
        forecasts[name] = evaluator.run(
            lambda s, g=with_garch: make_hybrid(seed=s, with_garch=g),
            name=name, seeds=SEEDS, refit_every=REFIT_EVERY, data=frame)
        print(f"  wall-clock: {evaluator.timings[name]:.1f}s", flush=True)

    os.makedirs(FC, exist_ok=True)
    for name, agg in forecasts.items():
        agg.to_csv(os.path.join(FC, f"{name}.csv"))

    metrics = _merge_metrics(evaluator.metrics())
    print("\n=== Metrics: new arms (QLIKE primary, mean +/- std over seeds) ===")
    new = metrics[metrics["model"].isin([n for n, _ in ARMS])]
    print(new[["model", "regime", "n", "qlike", "qlike_std", "rmse"]].to_string(
        index=False, formatters={"qlike": "{:.5f}".format, "qlike_std": "{:.5f}".format,
                                 "rmse": "{:.5f}".format}))

    dm = _recompute_dm_tests()
    print("\n=== Diebold-Mariano involving the hybrid (dm>0 => model_a worse) ===")
    rel = dm[(dm["model_a"].str.startswith("hybrid")) | (dm["model_b"].str.startswith("hybrid"))]
    print(rel.to_string(index=False, formatters={
        "dm_stat": "{:.3f}".format, "p_value": "{:.4g}".format}))

    print("\nNow run: python src/ablation_report.py   (adds the STRUCTURE effect)")


if __name__ == "__main__":
    main()
