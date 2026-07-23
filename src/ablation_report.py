"""
Ablation decomposition report (architecture vs information vs structure), computed from
saved forecasts.

Fast post-processing (no training): reads the seed-averaged forecasts and reports three
Diebold-Mariano comparisons per regime, decomposing where any deep-model gain comes from:

  INFORMATION effect  = full LSTM vs VIX-ablated LSTM
      (does adding VIX to the LSTM help?)  dm < 0 => full better => VIX helps
  ARCHITECTURE effect = HAR vs VIX-ablated LSTM  (both use only RV history)
      (does the LSTM architecture beat HAR on equal information?)  dm < 0 => HAR better
  STRUCTURE effect    = hybrid vs hybrid_nogarch  (identical but for the GARCH feature)
      (does an econometric structural input add value?)  dm < 0 => hybrid better

The structure effect is reported only when both hybrid arms are present (run
src/run_hybrid.py first); the first two effects are reported regardless, so this script
still works on the classical + LSTM results alone.

Both hybrid arms are trained on the same burn-in-trimmed frame, so their contrast isolates
the GARCH feature rather than training-sample size. That is also why `lstm` is NOT used as
the ablation arm for the structure effect: it trains on ~250 extra early rows. See
src/hybrid.py.

QLIKE values here are for the seed-averaged forecast of each model (so all models are
scored the same way and the DM tests are internally consistent). Per-seed mean +/- std is
in results/metrics_by_regime.csv.

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


def _exists(name):
    return os.path.exists(os.path.join(FC, f"{name}.csv"))


def main():
    har = _load("har")
    full = _load("lstm")
    novix = _load("lstm_novix")

    # The structure effect needs both hybrid arms; report it only once they exist.
    have_hybrid = _exists("hybrid") and _exists("hybrid_nogarch")
    hyb = _load("hybrid") if have_hybrid else None
    hyb_ng = _load("hybrid_nogarch") if have_hybrid else None
    if not have_hybrid:
        print("NOTE: hybrid forecasts not found; STRUCTURE effect omitted. "
              "Run src/run_hybrid.py to add it.\n")

    common = har.index.intersection(full.index).intersection(novix.index)
    if have_hybrid:
        common = common.intersection(hyb.index).intersection(hyb_ng.index)
    regimes = assign_regime(common)

    rows = []
    for regime in ["overall", *REGIME_ORDER]:
        idx = common if regime == "overall" else common[regimes == regime]
        if len(idx) < 2:
            continue
        lh = qlike_loss(har.loc[idx, "actual"], har.loc[idx, "forecast"])
        lf = qlike_loss(full.loc[idx, "actual"], full.loc[idx, "forecast"])
        ln = qlike_loss(novix.loc[idx, "actual"], novix.loc[idx, "forecast"])
        dm_info, p_info = diebold_mariano(lf, ln)     # full vs novix: dm<0 => VIX helps
        dm_arch, p_arch = diebold_mariano(lh, ln)     # har  vs novix: dm<0 => HAR better
        row = dict(
            regime=regime, n=len(idx),
            har_qlike=float(lh.mean()),
            lstm_full_qlike=float(lf.mean()),
            lstm_novix_qlike=float(ln.mean()),
            dm_info_full_vs_novix=dm_info, p_info=p_info,
            dm_arch_har_vs_novix=dm_arch, p_arch=p_arch)

        if have_hybrid:
            lhy = qlike_loss(hyb.loc[idx, "actual"], hyb.loc[idx, "forecast"])
            lng = qlike_loss(hyb_ng.loc[idx, "actual"], hyb_ng.loc[idx, "forecast"])
            # hybrid vs its matched arm: dm<0 => hybrid better => the GARCH feature helps
            dm_struct, p_struct = diebold_mariano(lhy, lng)
            row.update(hybrid_qlike=float(lhy.mean()),
                       hybrid_nogarch_qlike=float(lng.mean()),
                       dm_struct_hybrid_vs_nogarch=dm_struct, p_struct=p_struct)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ablation_decomposition.csv"), index=False)

    print("Ablation decomposition (QLIKE of seed-averaged forecasts)")
    print("  INFORMATION (VIX):    dm_info   < 0 => full better => VIX helps (removing it HURTS)")
    print("  ARCHITECTURE (LSTM):  dm_arch   < 0 => HAR beats the LSTM on RV-only info")
    if have_hybrid:
        print("  STRUCTURE (GARCH):    dm_struct < 0 => hybrid better => GARCH feature helps")
    print()
    fmt = {"har_qlike": "{:.5f}".format, "lstm_full_qlike": "{:.5f}".format,
           "lstm_novix_qlike": "{:.5f}".format,
           "dm_info_full_vs_novix": "{:.3f}".format, "p_info": "{:.4g}".format,
           "dm_arch_har_vs_novix": "{:.3f}".format, "p_arch": "{:.4g}".format,
           "hybrid_qlike": "{:.5f}".format, "hybrid_nogarch_qlike": "{:.5f}".format,
           "dm_struct_hybrid_vs_nogarch": "{:.3f}".format, "p_struct": "{:.4g}".format}
    print(out.to_string(index=False,
                        formatters={k: v for k, v in fmt.items() if k in out.columns}))


if __name__ == "__main__":
    main()
