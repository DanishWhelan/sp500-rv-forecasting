# CLAUDE.md — Project Context for Claude Code

## What this project is

MSc Data Science dissertation. An empirical comparison of classical and deep learning
models for forecasting **S&P 500 realised volatility**, under a rigorous, leakage-free
evaluation protocol. The contribution is **methodological, not architectural**: the point
is a fair, trustworthy comparison, not a new model.

Research question: do deep learning models improve realised-volatility forecast accuracy
over well-specified GARCH-family baselines, under leakage-free evaluation with proper
scoring rules and significance testing — and if so, does the gain come from architecture
or from exogenous information (VIX)?

**A null result is a valid, expected outcome.** Do not tune models to manufacture a deep-learning
win. The integrity of the comparison matters more than its direction.

## The model ladder (build in this order, simplest first)

1. EWMA (RiskMetrics) — naive baseline
2. GARCH(1,1) — core classical baseline
3. EGARCH and GJR-GARCH — asymmetric (REQUIRED: literature shows asymmetry is essential for equity data)
4. HAR-RV — strong realised-volatility benchmark (regression on daily/weekly/monthly RV averages)
5. LSTM — deep model
6. GARCH-LSTM hybrid — GARCH conditional-variance forecast fed as an LSTM input feature

## NON-NEGOTIABLE RULES (leakage prevention)

These are the whole point of the thesis. Violating any one invalidates the results.

- **Walk-forward evaluation only.** Expanding-origin. Never a random train/test split.
- **All scaling/normalisation fit on training data only**, at each origin. Never on the full series.
- **GARCH re-fit at each origin** on data up to that point. Never fit once on the whole series.
- **The GARCH feature for the hybrid must be the ONE-STEP-AHEAD FORECAST from data up to t-1**,
  never the in-sample fitted variance. This is the easiest leak to introduce and the most damaging.
- **Hyper-parameters selected without touching the test period.**
- If unsure whether something leaks, STOP and flag it rather than proceeding.

## Evaluation harness (build this BEFORE the models)

- Primary loss: **QLIKE** (robust to proxy noise; penalises under-prediction of vol).
- Also report: RMSE (for comparability).
- Significance: **Diebold-Mariano test** on every pairwise model comparison.
- Report deep-model results across **multiple random seeds** (mean ± std), not a single run.
- Evaluate **overall AND disaggregated by regime**: calm, 2008 GFC, COVID-19 (early 2020).
  NOTE: the RV target ends 2022-02-25 (Oxford-Man library discontinued), so the originally
  planned 2022-drawdown regime is dropped: only Jan-Feb 2022 is covered, too little to assess.
  Canonical regime windows live in `REGIMES` in `src/build_target.py`.
- Target: **realised volatility** (from Oxford-Man data), NOT squared returns.

## Ablation study (required by marking scheme)

For the LSTM/hybrid, measure the contribution of each component by removing it:
- Remove VIX feature → does accuracy drop? (isolates value of exogenous information)
- Remove GARCH feature from hybrid → (isolates value of econometric structure)
This decomposes any gain into architecture vs information. It is a core contribution.

## Data

- `data/raw/spy_daily.csv` — SPY OHLCV (yfinance). Immutable.
- `data/raw/vix_daily.csv` — VIX (yfinance). Exogenous feature. Immutable.
- `data/raw/oxfordman_raw.csv` — FULL archived Oxford-Man realised library (all symbols, all
  estimators). Immutable. build_target.py filters to symbol `.SPX` and column `rv5`. Ends 2022-02-25.
- `data/processed/modelling_data.csv` — built by `src/build_target.py`. What models consume.
- **NEVER edit anything in data/raw/ by hand.** All cleaning is in code, regenerable from raw.

## Reproducibility

- Every number in the dissertation must be regenerable from data/raw by running code.
- Set random seeds explicitly. Save results to `results/` as CSV, not just printed.
- Commit at the end of each working session.

## Validation requirement (from supervisor)

The GARCH(1,1) baseline must be validated against published S&P 500 parameter estimates
to confirm correct specification before any comparison is drawn. A misspecified baseline
invalidates everything. When GARCH is fitted, report the parameters and compare to literature.

## Code style

- Clear, readable, commented where the finance/stats logic is non-obvious.
- Prose in comments/docstrings: no em-dashes.
- Prefer explicit, auditable code over clever one-liners — this is research code that must be
  trusted and reproduced, not production code optimised for elegance.
- Scripts in src/, exploration in notebooks/, outputs to results/.

## Current status

Week 1 scaffold complete: download_data.py, build_target.py, eda.py all written and tested.
Next: Week 2 — the evaluation harness, then the classical models (EWMA, GARCH, EGARCH, GJR, HAR-RV).
