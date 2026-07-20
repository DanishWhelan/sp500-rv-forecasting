# Week 1 Checklist — Foundations & Data

Goal: a clean, frozen, validated modelling dataframe. NO modelling this week.

## Day 1 (Mon) — Environment + supervisor
- [ ] Fix Anaconda/PowerShell PATH. `python --version` and `conda --version` both work in a fresh terminal.
- [ ] `conda create -n thesis python=3.11 -y && conda activate thesis`
- [ ] `pip install -r requirements.txt`
- [ ] Verify: `python -c "import pandas,numpy,arch,yfinance,statsmodels; print('ok')"`

## Day 2 (Tue) — Repo scaffolding
- [ ] `git init`; commit this folder structure.
- [ ] Confirm .gitignore excludes data/raw and data/processed contents.
- [ ] First commit. Commit at the end of every day from here.

## Day 3 (Wed) — Download & freeze raw data
- [ ] `python src/download_data.py`  (creates spy_daily.csv, vix_daily.csv)
- [ ] Download the full archived Oxford-Man library -> data/raw/oxfordman_raw.csv (Wayback mirror; see README provenance table for the exact URL). build_target.py filters it to .SPX / rv5.
- [ ] Fill in README provenance table: URLs, today's date, Oxford-Man version.
- [ ] Commit. RULE: never edit anything in data/raw/ by hand again.

## Day 4 (Thu) — Build & validate the RV target  (most delicate step)
- [ ] `python src/build_target.py`
- [ ] Check the crisis output: COVID-2020 and 2022 should both print "spikes as expected".
      If either says CHECK, your date alignment is wrong — fix before proceeding.
- [ ] Confirm data/processed/modelling_data.csv exists and columns look right.

## Day 5 (Fri) — Exploratory analysis
- [ ] `python src/eda.py`
- [ ] Inspect results/fig3_acf.png — returns ACF ~0, squared-returns ACF strongly positive.
      This is the figure that justifies GARCH. It goes in section 4.4.
- [ ] Read summary_stats.csv; note excess kurtosis (fat tails) and the two ACF(1) values.
- [ ] Write ~1 page data description now, while fresh -> feeds dissertation section 4.4.

## End-of-week gate (all four before Week 2)
- [ ] Environment works in a fresh terminal.
- [ ] Repo committed; raw data frozen and logged in README.
- [ ] Single clean data/processed/modelling_data.csv (returns + RV target + VIX).
- [ ] RV target sanity-checked against crisis dates; you understand what it represents.

## If a day slips
Protect Days 3-4 above all. Day 5 can compress to a couple of hours. Day 1's
environment fix cannot slip — escalate immediately if it fights you.
