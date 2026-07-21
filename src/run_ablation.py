"""
VIX ablation for the LSTM (architecture vs information decomposition).

Re-runs the LSTM with ONLY lagged log realised variance as input (VIX removed), under the
same protocol as the full model (monthly refit, 5 seeds), and compares to the full LSTM
(log-RV + VIX). Interpretation:
  - if removing VIX significantly DEGRADES accuracy -> the exogenous VIX information was
    contributing, and the HAR-parity came partly from information;
  - if removing VIX does NOT change accuracy -> the parity came from architecture / lagged
    realised volatility alone (the same signal HAR uses), not from VIX.

Does NOT overwrite the canonical metrics/DM tables; writes only the ablated forecasts and a
dedicated ablation summary (results/ablation_vix.csv).

Usage: python src/run_ablation.py
"""
import os

import numpy as np
import pandas as pd

from harness import (WalkForwardEvaluator, qlike_loss, diebold_mariano,
                     assign_regime, REGIME_ORDER)
from lstm import LSTMModel

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "processed", "modelling_data.csv")
RES = os.path.join(BASE, "results")
FULL_FC = os.path.join(RES, "forecasts", "lstm.csv")
METRICS = os.path.join(RES, "metrics_by_regime.csv")

SEEDS = (0, 1, 2, 3, 4)
REFIT_EVERY = 21


def main():
    data = pd.read_csv(DATA, index_col=0, parse_dates=True)
    if not os.path.exists(FULL_FC):
        raise SystemExit("Full LSTM forecasts not found; run src/run_evaluation.py first.")
    full = pd.read_csv(FULL_FC, index_col=0, parse_dates=True)
    # full LSTM mean +/- std (mean over 5 seeds) from the saved metrics table
    full_metrics = pd.read_csv(METRICS)
    full_lstm = full_metrics[full_metrics["model"] == "lstm"].set_index("regime")

    ev = WalkForwardEvaluator(data)
    print("Running VIX-ablated LSTM (features=realised_variance only), "
          "5 seeds, refit_every=21 ...", flush=True)
    novix = ev.run(lambda s: LSTMModel(seed=s, features=("realised_variance",),
                                       name="lstm_novix"),
                   name="lstm_novix", seeds=SEEDS, refit_every=REFIT_EVERY)
    print(f"  wall-clock: {ev.timings['lstm_novix']:.1f}s", flush=True)

    os.makedirs(os.path.join(RES, "forecasts"), exist_ok=True)
    novix.to_csv(os.path.join(RES, "forecasts", "lstm_novix.csv"))
    novix_metrics = ev.metrics().set_index("regime")   # ablated model mean +/- std over seeds

    # per-regime comparison; DM on the seed-averaged forecasts of full vs ablated
    common = full.index.intersection(novix.index)
    regimes = assign_regime(common)
    rows = []
    for regime in ["overall", *REGIME_ORDER]:
        idx = common if regime == "overall" else common[regimes == regime]
        if len(idx) < 2:
            continue
        la = qlike_loss(full.loc[idx, "actual"], full.loc[idx, "forecast"])
        lb = qlike_loss(novix.loc[idx, "actual"], novix.loc[idx, "forecast"])
        dm, p = diebold_mariano(la, lb)   # dm>0 => full worse (VIX removal helps); <0 => hurts
        rows.append(dict(
            regime=regime, n=len(idx),
            full_qlike=float(full_lstm.loc[regime, "qlike"]),
            full_std=float(full_lstm.loc[regime, "qlike_std"]),
            novix_qlike=float(novix_metrics.loc[regime, "qlike"]),
            novix_std=float(novix_metrics.loc[regime, "qlike_std"]),
            delta=float(novix_metrics.loc[regime, "qlike"]) - float(full_lstm.loc[regime, "qlike"]),
            dm_full_vs_novix=dm, p_value=p))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RES, "ablation_vix.csv"), index=False)

    print("\n=== VIX ablation: full LSTM (log-RV + VIX) vs ablated (log-RV only) ===")
    print("delta = novix_qlike - full_qlike  (positive => removing VIX HURTS)")
    print("dm_full_vs_novix > 0 => full worse (VIX removal helps); < 0 => VIX removal hurts\n")
    print(summary.to_string(index=False, formatters={
        "full_qlike": "{:.5f}".format, "full_std": "{:.5f}".format,
        "novix_qlike": "{:.5f}".format, "novix_std": "{:.5f}".format,
        "delta": "{:+.5f}".format, "dm_full_vs_novix": "{:.3f}".format,
        "p_value": "{:.4f}".format}))


if __name__ == "__main__":
    main()
