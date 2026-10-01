"""
Day 5: Exploratory data analysis.

Generates the figures that motivate the modelling choices and go into the
dissertation's data-description section (4.4). Each figure earns its place by
justifying a later decision.

Figures produced (into results/):
  fig1_returns.png              returns over time
  fig2_clustering.png           squared returns -> volatility clustering
  fig3_acf.png                  ACF of returns vs ACF of squared returns (the key one)
  fig4_distribution.png         return distribution vs normal (fat tails)
  fig5_vix_vs_rv.png            VIX overlaid on realised volatility

Usage:
    python src/eda.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "modelling_data.csv"), index_col=0, parse_dates=True)
r = df["log_return"]

# --- Fig 1: returns over time ---
plt.figure(figsize=(10, 3))
plt.plot(df.index, r, lw=0.5)
plt.title("SPY daily log returns")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig1_returns.png"), dpi=120); plt.close()

# --- Fig 2: squared returns (volatility clustering) ---
plt.figure(figsize=(10, 3))
plt.plot(df.index, r**2, lw=0.5, color="firebrick")
plt.title("Squared daily returns: volatility clustering")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig2_clustering.png"), dpi=120); plt.close()

# --- Fig 3: ACF of returns vs ACF of squared returns (THE key figure) ---
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
plot_acf(r.dropna(), lags=40, ax=ax[0], title="ACF of returns")
plot_acf((r**2).dropna(), lags=40, ax=ax[1], title="ACF of squared returns")
fig.suptitle("Returns are near-uncorrelated; squared returns are strongly autocorrelated",
             fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig3_acf.png"), dpi=120); plt.close()

# --- Fig 4: distribution vs normal (fat tails) ---
plt.figure(figsize=(6, 4))
stats.probplot(r.dropna(), dist="norm", plot=plt)
plt.title("Normal Q-Q plot of returns: departure in tails indicates fat tails")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig4_distribution.png"), dpi=120); plt.close()

# --- Fig 5: VIX vs realised volatility ---
plt.figure(figsize=(10, 3.2))
plt.plot(df.index, df["rv_annualised"], lw=0.8, label="Realised volatility (annualised)")
plt.plot(df.index, df["vix"] / 100, lw=0.8, alpha=0.8, label="VIX / 100")
plt.legend(); plt.title("VIX vs realised volatility")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig5_vix_vs_rv.png"), dpi=120); plt.close()

# --- summary statistics for the data-description section ---
desc = {
    "n_observations": len(r),
    "start": str(df.index.min().date()),
    "end": str(df.index.max().date()),
    "mean_daily_return": float(r.mean()),
    "std_daily_return": float(r.std()),
    "skewness": float(stats.skew(r.dropna())),
    "excess_kurtosis": float(stats.kurtosis(r.dropna())),  # 0 = normal
    "acf1_returns": float(r.autocorr(1)),
    "acf1_squared_returns": float((r**2).autocorr(1)),
}
pd.Series(desc).to_csv(os.path.join(RES, "summary_stats.csv"))
print("Figures and summary_stats.csv written to results/")
for k, v in desc.items():
    print(f"  {k}: {v}")
