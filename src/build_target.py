"""
Day 4: Build and validate the realised-volatility target.

Loads raw SPY, VIX and Oxford-Man realised-variance data; computes log returns;
aligns everything to a single daily dataframe; converts realised variance to
realised volatility (annualised); runs sanity checks; and saves the processed
modelling dataframe.

This is the most methodologically delicate step in Week 1: every downstream model
consumes data/processed/modelling_data.csv, so alignment errors here contaminate
the whole study.

Usage:
    python src/build_target.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

TRADING_DAYS = 252

# --- Oxford-Man realised-library selection (CHANGE THESE to switch target measure) ---
# The raw mirror data/raw/oxfordman_raw.csv is the FULL, immutable Oxford-Man file:
# every index symbol stacked together, with many realised-variance estimators per row.
# We filter to one symbol and pick one estimator column explicitly, in code, so the raw
# file stays untouched and the choice is auditable.
OXFORDMAN_FILE = "oxfordman_raw.csv"   # full unfiltered archived mirror (immutable)
OXFORDMAN_SYMBOL = ".SPX"              # S&P 500 index symbol in the Oxford-Man library
RV_COLUMN = "rv5"                      # 5-minute realised VARIANCE. Alternatives: rv5_ss,
                                       # rk_parzen, bv, medrv, rv10, ... (see file header)

# --- Evaluation regimes (canonical definition; the harness imports these) ---
# Crisis windows for regime-disaggregated evaluation. "calm" is the complement: any date
# in the sample not inside one of these windows. The originally planned 2022-drawdown regime
# was DROPPED because the Oxford-Man RV target ends 2022-02-25 (library discontinued), leaving
# only Jan-Feb 2022 covered, too little to evaluate. See README limitation note.
REGIMES = {
    "gfc_2008":   ("2008-09-01", "2009-03-31"),  # Lehman collapse through the March-2009 bottom
    "covid_2020": ("2020-02-20", "2020-04-30"),  # COVID crash and acute-volatility window
}


def load_price(fname, col="Close"):
    # yfinance 1.5.x writes a 3-row header for single-ticker downloads:
    #   row 1: Price,Adj Close,Close,High,Low,Open,Volume
    #   row 2: Ticker,SPY,SPY,...          <- leaks into the body if read naively
    #   row 3: Date,,,,,,                  <- ditto
    # Read flat, then coerce: parse the first column as dates and drop any rows whose
    # "date" is not a real date (i.e. the stray Ticker/Date header rows -> NaT). This
    # also handles the older flat-header format, where nothing gets dropped.
    raw = pd.read_csv(os.path.join(RAW, fname))
    if col not in raw.columns:
        raise SystemExit(f"ERROR: column {col!r} not in {fname}; columns={list(raw.columns)}")
    date_col = raw.columns[0]
    # yfinance writes ISO dates (YYYY-MM-DD); an explicit format avoids the slow per-element
    # dateutil fallback (and its warning). The stray Ticker/Date header rows fail the format
    # and become NaT, which is exactly what we drop below.
    idx = pd.to_datetime(raw[date_col], format="%Y-%m-%d", errors="coerce")
    vals = pd.to_numeric(raw[col], errors="coerce")
    mask = idx.notna().to_numpy()  # keep only real date rows (drops Ticker/Date header rows)
    out = pd.Series(vals.to_numpy()[mask], index=idx.to_numpy()[mask], name=col)
    return out.sort_index()


def main():
    # --- 1. SPY close -> log returns ---
    close = load_price("spy_daily.csv", "Close")
    log_ret = np.log(close).diff().rename("log_return")

    # --- 2. VIX close (exogenous feature) ---
    vix = load_price("vix_daily.csv", "Close").rename("vix")

    # --- 3. Oxford-Man realised variance -> realised volatility ---
    # Read the full raw mirror (all symbols, all estimators) and select explicitly.
    ox = pd.read_csv(os.path.join(RAW, OXFORDMAN_FILE), index_col=0, parse_dates=True)

    if "Symbol" not in ox.columns:
        raise SystemExit(f"ERROR: no 'Symbol' column in {OXFORDMAN_FILE}; is this the full raw mirror?")
    symbols = sorted(ox["Symbol"].unique())
    if OXFORDMAN_SYMBOL not in symbols:
        raise SystemExit(f"ERROR: symbol {OXFORDMAN_SYMBOL!r} not found. Available: {symbols}")
    if RV_COLUMN not in ox.columns:
        raise SystemExit(f"ERROR: RV column {RV_COLUMN!r} not found. Available: {list(ox.columns)}")

    # Filter to the chosen index symbol, then take the chosen realised-variance column.
    ox = ox[ox["Symbol"] == OXFORDMAN_SYMBOL].copy()
    # The raw index is tz-aware (UTC) and parses as an object dtype; coerce to a clean
    # tz-naive DatetimeIndex so it aligns with the tz-naive SPY/VIX price dates below.
    ox.index = pd.to_datetime(ox.index, utc=True).tz_localize(None).normalize()
    ox = ox[~ox.index.duplicated(keep="first")].sort_index()

    rvar = pd.to_numeric(ox[RV_COLUMN], errors="coerce").rename("realised_variance")
    # This print is the audit trail: confirm the right symbol and column were selected.
    print(f"Oxford-Man selection: symbol={OXFORDMAN_SYMBOL!r}  column={RV_COLUMN!r}  "
          f"rows={len(rvar)}  ({ox.index.min().date()} to {ox.index.max().date()})")

    # realised volatility (daily, then annualised for interpretability)
    rvol_daily = np.sqrt(rvar).rename("rv_daily")
    rvol_ann = (rvol_daily * np.sqrt(TRADING_DAYS)).rename("rv_annualised")

    # --- 4. align on common trading days (inner join) ---
    df = pd.concat([close.rename("close"), log_ret, vix, rvar, rvol_daily, rvol_ann],
                   axis=1).dropna()
    print(f"Aligned dataframe: {len(df)} rows, "
          f"{df.index.min().date()} to {df.index.max().date()}")

    # --- 5. sanity checks (fail loudly if alignment is wrong) ---
    assert df.index.is_monotonic_increasing, "dates not sorted"
    assert (df["rv_daily"] > 0).all(), "non-positive realised volatility present"
    assert df["log_return"].abs().max() < 0.5, "implausible daily return (>50%), check data"

    # crisis check: RV should spike in each defined crisis regime, and each window must
    # actually be covered by the sample (guards against a silently truncated RV target).
    for label, (lo, hi) in REGIMES.items():
        window = df.loc[lo:hi, "rv_annualised"]
        assert len(window), f"regime {label!r} ({lo} to {hi}) has no data. RV sample truncated?"
        ratio = window.max() / df["rv_annualised"].median()
        print(f"  {label} ({lo} to {hi}): {len(window)} days, peak RV = {ratio:.1f}x median "
              f"({'OK, spikes as expected' if ratio > 1.5 else 'CHECK: no clear spike'})")

    # --- 6. save processed modelling dataframe ---
    out = os.path.join(PROC, "modelling_data.csv")
    df.to_csv(out)
    print(f"\nSaved -> {out}")
    print("Columns:", list(df.columns))
    print("\nHead:\n", df.head(3).to_string())


if __name__ == "__main__":
    main()
