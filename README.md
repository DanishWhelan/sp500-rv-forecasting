# Beyond GARCH? — Volatility Forecasting Comparison

MSc Data Science dissertation. Empirical comparison of classical (GARCH-family, HAR-RV)
and deep learning (LSTM, GARCH-LSTM hybrid) models for forecasting S&P 500 realised
volatility, under a leakage-free walk-forward protocol with QLIKE and Diebold-Mariano testing.

## Environment setup

```bash
conda create -n vol python=3.11 -y
conda activate vol
pip install -r requirements.txt
```

The pinned numerical stack (numpy/scipy/arch/statsmodels) only ships wheels for
Python 3.11/3.12, so the 3.11 env above is required; newer interpreters try to
build from source and fail.

Verify the environment works in a fresh terminal:

```bash
python --version      # should print Python 3.11.x
python -c "import pandas, numpy, arch, yfinance, statsmodels; print('ok')"
```

## Project structure

```
data/raw/         Immutable downloaded data. NEVER edit by hand.
data/processed/   Cleaned, aligned modelling dataframe (built by code).
notebooks/        Exploratory analysis.
src/              Reusable scripts (data loading, models, evaluation harness).
results/          Output tables and figures.
```

## Reproducibility rule

Every number in the dissertation must be regenerable from `data/raw` by running code.
Raw files are never edited by hand. All cleaning happens in `src/`.

## Data provenance log

Fill this in as you download. This is what makes the archived data citable.

| Dataset | File | Source URL | Downloaded | Notes |
|---|---|---|---|---|
| SPY daily OHLCV | data/raw/spy_daily.csv | yfinance 1.5.1 (ticker SPY) | 2026-07-20 | full history, 2000-01-03 to 2026-07-20 (6675 rows) |
| VIX daily | data/raw/vix_daily.csv | yfinance 1.5.1 (ticker ^VIX) | 2026-07-20 | exogenous feature, 2000-01-03 to 2026-07-20 |
| S&P 500 realised volatility | data/raw/oxfordman_raw.csv | https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip | 2026-07-20 | Full unfiltered Oxford-Man library (Wayback snapshot 2022-03-01). Library discontinued; this is the frozen archived copy. S&P 500 = symbol `.SPX`, target measure `rv5`, filtered in build_target.py. Data ends 2022-02-25. Cite Heber et al. (2009) |

**yfinance version note:** requirements.txt pinned `yfinance==0.2.40`, which no longer
authenticates against Yahoo's current API (empty response / `No timezone found`). It was
upgraded to `1.5.1` to perform the one-time download and the pin relaxed to `yfinance>=1.5`.
yfinance is only the download pipe: once `data/raw/*.csv` are frozen it plays no further
role, so this does not affect the numerical reproducibility contract.

**RV coverage limitation:** because the Oxford-Man library was frozen at discontinuation,
the realised-volatility target ends **2022-02-25**. The processed modelling set
(`data/processed/modelling_data.csv`, 4877 rows) is therefore 2000-01-04 to 2022-02-25,
even though SPY/VIX run to 2026. This materially affects the planned **2022-drawdown regime**:
only Jan-Feb 2022 is covered (the drawdown continued through Oct 2022). See below.

## Progress

- [ ] Day 1: environment + repo
- [ ] Day 2: git scaffolding committed
- [ ] Day 3: raw data downloaded and frozen
- [ ] Day 4: realised-volatility target built and validated
- [ ] Day 5: exploratory analysis + data description written
