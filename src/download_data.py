"""
Day 3: Download and freeze raw data.

Downloads SPY and VIX daily data via yfinance and saves UNTOUCHED to data/raw/.
Run once. Do not edit the outputs by hand.

The Oxford-Man realised volatility data must be downloaded manually from the archived
Wayback mirror (see README provenance log for the exact URL) and placed, UNZIPPED, at
data/raw/oxfordman_raw.csv. This is the full library (all symbols, all estimators); it is
not available via an API. build_target.py filters it to symbol .SPX and column rv5.

Usage:
    python src/download_data.py
"""
import os
from datetime import date
import yfinance as yf

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW, exist_ok=True)


def download(ticker, filename, start="2000-01-01"):
    print(f"Downloading {ticker} ...")
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise SystemExit(f"ERROR: no data returned for {ticker}. Check ticker / network.")
    path = os.path.join(RAW, filename)
    df.to_csv(path)
    print(f"  saved {len(df)} rows -> {path}")
    print(f"  range: {df.index.min().date()} to {df.index.max().date()}")
    return df


if __name__ == "__main__":
    print(f"Download date: {date.today().isoformat()}  (record this in README)\n")
    download("SPY", "spy_daily.csv")
    download("^VIX", "vix_daily.csv")
    print("\nDone. Now:")
    print("  1. Record today's date and tickers in README provenance log.")
    print("  2. Manually download + unzip the Oxford-Man library to data/raw/oxfordman_raw.csv.")
    print("  3. Do NOT edit any file in data/raw/ by hand from here on.")
