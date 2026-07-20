"""
Plot a model's one-step-ahead forecasts against realised volatility over the test
period. Reads results/forecasts/{name}.csv and saves results/fig_{name}_vs_rv.png.

Volatility is shown ANNUALISED in percent (sqrt of variance x sqrt(252) x 100) for
interpretability, and the crisis regimes are shaded so tracking vs lagging is visible.

Usage:
    python src/plot_forecasts.py [model_name]      # default: ewma
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


def annualised_vol_pct(variance):
    """Daily realised VARIANCE -> annualised volatility in percent."""
    return np.sqrt(variance) * np.sqrt(252) * 100.0


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


if __name__ == "__main__":
    plot_model(sys.argv[1] if len(sys.argv) > 1 else "ewma")
