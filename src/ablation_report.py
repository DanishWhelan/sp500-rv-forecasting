"""
Ablation decomposition report (architecture vs information), computed from saved forecasts.

Fast post-processing (no training): reads the seed-averaged forecasts of HAR, the full LSTM
(log-RV + VIX) and the VIX-ablated LSTM (log-RV only), and reports two Diebold-Mariano
comparisons per regime:

  INFORMATION effect  = full LSTM vs VIX-ablated LSTM
      (does adding VIX to the LSTM help?)  dm < 0 => full better => VIX helps
  ARCHITECTURE effect = HAR vs VIX-ablated LSTM  (both use only RV history)
      (does the LSTM architecture beat HAR on equal information?)  dm < 0 => HAR better

QLIKE values here are for the seed-averaged forecast of each model (so all three are scored
the same way and the DM tests are internally consistent). Per-seed mean +/- std for the LSTM
variants is in results/ablation_vix.csv / results/metrics_by_regime.csv.

Usage: python src/ablation_report.py
"""
import os

import pandas as pd

from harness import qlike_loss, diebold_mariano, assign_regime, REGIME_ORDER

BASE = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(BASE, "results")
FC = os.path.join(RES, "forecasts")


def _load(name):
    return pd.read_csv(os.path.join(FC, f"{name}.csv"), index_col=0, parse_dates=True)


def main():
    har = _load("har")
    full = _load("lstm")
    novix = _load("lstm_novix")

    common = har.index.intersection(full.index).intersection(novix.index)
    regimes = assign_regime(common)

    rows = []
    for regime in ["overall", *REGIME_ORDER]:
        idx = common if regime == "overall" else common[regimes == regime]
        if len(idx) < 2:
            continue
        lh = qlike_loss(har.loc[idx, "actual"], har.loc[idx, "forecast"])
        lf = qlike_loss(full.loc[idx, "actual"], full.loc[idx, "forecast"])
        ln = qlike_loss(novix.loc[idx, "actual"], novix.loc[idx, "forecast"])
        dm_info, p_info = diebold_mariano(lf, ln)     # full vs novix: dm>0 => VIX helps
        dm_arch, p_arch = diebold_mariano(lh, ln)     # har  vs novix: dm<0 => HAR better
        rows.append(dict(
            regime=regime, n=len(idx),
            har_qlike=float(lh.mean()),
            lstm_full_qlike=float(lf.mean()),
            lstm_novix_qlike=float(ln.mean()),
            dm_info_full_vs_novix=dm_info, p_info=p_info,
            dm_arch_har_vs_novix=dm_arch, p_arch=p_arch))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ablation_decomposition.csv"), index=False)

    print("Ablation decomposition (QLIKE of seed-averaged forecasts)")
    print("  INFORMATION (VIX):    dm_info  < 0  => full better => VIX helps (removing it HURTS)")
    print("  ARCHITECTURE (LSTM):  dm_arch  < 0  => HAR beats the LSTM on RV-only info\n")
    print(out.to_string(index=False, formatters={
        "har_qlike": "{:.5f}".format, "lstm_full_qlike": "{:.5f}".format,
        "lstm_novix_qlike": "{:.5f}".format,
        "dm_info_full_vs_novix": "{:.3f}".format, "p_info": "{:.4f}".format,
        "dm_arch_har_vs_novix": "{:.3f}".format, "p_arch": "{:.4f}".format}))


if __name__ == "__main__":
    main()
