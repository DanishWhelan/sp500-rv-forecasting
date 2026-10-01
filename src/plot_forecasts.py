"""
Plot one-step-ahead forecasts against realised volatility over the test period.

Two entry points, both reading results/forecasts/{name}.csv (written by the walk-forward
harness) and writing PNGs to results/:

  plot_model(name)                 one model vs realised volatility
  plot_comparison(names, out)      several models vs realised volatility, optionally
                                   zoomed to a single regime window

Volatility is shown ANNUALISED in percent (sqrt of variance x sqrt(252) x 100) for
interpretability, and the crisis regimes are shaded so tracking vs lagging is visible.

This module is PLOTTING ONLY. It never refits a model and never writes a results CSV:
it reads the stored forecasts as given, so figures cannot disagree with the tables.

Usage:
    python src/plot_forecasts.py [model_name]      # default: ewma, single model
    python src/plot_forecasts.py --comparison      # the three comparison figures
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import REGIMES

BASE = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(BASE, "results")

# Colourblind-safe categorical hues (Okabe-Ito). Assigned to models in a FIXED order and
# never cycled, so a model keeps its colour across every figure in the dissertation.
# Checked with the palette validator: worst adjacent pair is deltaE 11.0 under deuteranopia
# and 25.8 under normal vision, all three above the 3:1 contrast floor. The matplotlib
# default trio was rejected here: its green and orange collapse to deltaE 0.7 under
# protanopia, i.e. they are the same colour to a red-blind reader in a printed thesis.
MODEL_COLOURS = {
    "har":            "#0072B2",  # blue
    "egarch":         "#009E73",  # bluish green
    "lstm":           "#D55E00",  # vermillion
    "garch":          "#CC79A7",  # reddish purple
    "gjr":            "#7A5195",  # purple
    "ewma":           "#B8860B",  # dark goldenrod
    "hybrid":         "#56B4E9",  # sky blue
    "hybrid_nogarch": "#8C6D31",  # brown
    "lstm_novix":     "#444444",  # dark grey
}
ACTUAL_COLOUR = "0.35"  # realised volatility is a neutral reference series, not a category

# Readable names for the regime keys, for figure titles and shading labels. The keys
# themselves stay canonical (REGIMES in build_target.py); this is presentation only.
REGIME_LABELS = {
    "gfc_2008": "Global financial crisis, 2008-09",
    "covid_2020": "COVID-19 crash, 2020",
}

# Display names for the legend, so the figure reads as the dissertation does.
MODEL_LABELS = {
    "har": "HAR-RV",
    "egarch": "EGARCH",
    "lstm": "LSTM",
    "garch": "GARCH(1,1)",
    "gjr": "GJR-GARCH",
    "ewma": "EWMA",
    "hybrid": "GARCH-LSTM hybrid",
    "hybrid_nogarch": "LSTM (no GARCH feature)",
    "lstm_novix": "LSTM (no VIX)",
}


def annualised_vol_pct(variance):
    """Daily realised VARIANCE -> annualised volatility in percent."""
    return np.sqrt(variance) * np.sqrt(252) * 100.0


def load_forecast(name):
    """Read one model's stored walk-forward forecasts. Read-only."""
    path = os.path.join(RES, "forecasts", f"{name}.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def label_of(name):
    return MODEL_LABELS.get(name, name.upper())


def plot_model(name="ewma"):
    path = os.path.join(RES, "forecasts", f"{name}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, annualised_vol_pct(df["actual"]), lw=0.7, color="0.35",
            label="Realised volatility")
    ax.plot(df.index, annualised_vol_pct(df["forecast"]), lw=0.7, color="tab:orange",
            alpha=0.9, label=f"{name.upper()} forecast")

    # shade and label the crisis regimes
    ymax = annualised_vol_pct(df[["actual", "forecast"]].to_numpy()).max()
    for rname, (lo, hi) in REGIMES.items():
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="tab:red", alpha=0.08)
        ax.text(pd.Timestamp(lo), ymax * 0.98, rname, fontsize=8, color="tab:red",
                va="top", ha="left")

    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title(f"{name.upper()} one-step-ahead forecast vs realised volatility "
                 f"({df.index.min().date()} to {df.index.max().date()})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.margins(x=0)
    fig.tight_layout()

    out = os.path.join(RES, f"fig_{name}_vs_rv.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_comparison(names, out_name, zoom=None, pad_days=45, title=None):
    """
    Plot several models' forecasts against realised volatility on ONE axis.

    names     list of model names, i.e. stems of results/forecasts/{name}.csv
    out_name  output file stem, saved to results/{out_name}.png
    zoom      None for the full test period, or a key of REGIMES to zoom to that window
    pad_days  calendar days of context shown either side of a zoomed regime, so the
              run-up and the decay are visible rather than just the spike. The regime
              itself stays shaded, so the scored window is never ambiguous.

    Everything is drawn in annualised volatility percent on a single y-axis: these are
    all forecasts of the same quantity, so a second axis would be meaningless here.
    """
    frames = {n: load_forecast(n) for n in names}

    # The harness writes the same target column into every forecast file. Assert it
    # rather than assume it: if two models were scored on different samples, overlaying
    # them would be comparing different things and the figure would quietly mislead.
    reference = frames[names[0]]
    for n, df in frames.items():
        if not df.index.equals(reference.index):
            raise ValueError(f"{n} is scored on a different date index than {names[0]}")
        if not np.allclose(df["actual"], reference["actual"], rtol=0, atol=0):
            raise ValueError(f"{n} has a different 'actual' column than {names[0]}")

    # Restrict to the plotted window.
    if zoom is not None:
        lo, hi = REGIMES[zoom]
        view_lo = pd.Timestamp(lo) - pd.Timedelta(days=pad_days)
        view_hi = pd.Timestamp(hi) + pd.Timedelta(days=pad_days)
    else:
        view_lo, view_hi = reference.index.min(), reference.index.max()
    frames = {n: df.loc[view_lo:view_hi] for n, df in frames.items()}
    reference = frames[names[0]]
    if reference.empty:
        raise ValueError(f"no observations in the requested window {view_lo} to {view_hi}")

    # Zoomed views have far fewer points, so thicker lines stay readable in print.
    lw_actual, lw_model = (0.7, 0.7) if zoom is None else (1.6, 1.4)
    figsize = (12, 4) if zoom is None else (10, 4.2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(reference.index, annualised_vol_pct(reference["actual"]),
            lw=lw_actual, color=ACTUAL_COLOUR, label="Realised volatility", zorder=2)
    for n in names:
        ax.plot(frames[n].index, annualised_vol_pct(frames[n]["forecast"]),
                lw=lw_model, color=MODEL_COLOURS.get(n, "black"), alpha=0.9,
                label=f"{label_of(n)} forecast", zorder=3)

    # Shade the crisis regimes that are visible in this window, and label them only on
    # the full-period figure (on a zoom the title already names the window).
    stacked = np.concatenate(
        [annualised_vol_pct(reference["actual"].to_numpy())]
        + [annualised_vol_pct(frames[n]["forecast"].to_numpy()) for n in names]
    )
    ymax = np.nanmax(stacked)
    # Headroom above the data so the regime captions sit clear of every line.
    ax.set_ylim(bottom=0, top=ymax * 1.14)
    for rname, (lo, hi) in REGIMES.items():
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        if hi_ts < view_lo or lo_ts > view_hi:
            continue
        ax.axvspan(lo_ts, hi_ts, color="tab:red", alpha=0.08, zorder=1)
        if zoom is None:
            ax.text(lo_ts, ymax * 1.05, REGIME_LABELS.get(rname, rname), fontsize=8,
                    color="tab:red", va="center", ha="center")

    if title is None:
        span = f"{reference.index.min().date()} to {reference.index.max().date()}"
        if zoom is None:
            title = f"One-step-ahead forecasts vs realised volatility ({span})"
        else:
            title = (f"One-step-ahead forecasts vs realised volatility: "
                     f"{REGIME_LABELS.get(zoom, zoom)} ({span})")

    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title(title)
    # On the full-period figure the top right is where the COVID spike and its caption
    # live, so the legend goes left, where the early-sample data is quiet.
    ax.legend(loc="upper left" if zoom is None else "upper right",
              framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.15, lw=0.5)          # recessive grid, behind the data
    ax.set_axisbelow(True)
    ax.margins(x=0)
    fig.tight_layout()

    out = os.path.join(RES, f"{out_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}  ({len(reference)} observations)")
    return out


# The headline comparison for the results chapter: best classical (HAR-RV), best
# asymmetric GARCH (EGARCH), and the deep model (LSTM), on identical scored samples.
COMPARISON_MODELS = ["har", "egarch", "lstm"]


def make_comparison_figures():
    """Build the three results-chapter comparison figures."""
    outs = [
        plot_comparison(COMPARISON_MODELS, "fig_comparison_full"),
        plot_comparison(COMPARISON_MODELS, "fig_comparison_covid2020", zoom="covid_2020"),
        plot_comparison(COMPARISON_MODELS, "fig_comparison_gfc2008", zoom="gfc_2008"),
    ]
    return outs


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "ewma"
    if arg == "--comparison":
        make_comparison_figures()
    else:
        plot_model(arg)
