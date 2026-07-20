"""Shared pytest fixtures for the harness self-tests.

Uses a synthetic, deterministic dataset (not the real modelling_data.csv) so the tests
are self-contained, fast, and independent of the data pipeline. The synthetic frame
spans 2000-2021 business days so it covers both crisis-regime windows.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make src/ importable as top-level modules (harness, model_base, build_target).
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model_base import VolatilityModel  # noqa: E402


@pytest.fixture
def sample_data():
    """A positive, well-behaved variance series with returns and a VIX-like feature."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", "2021-12-31")
    n = len(idx)
    rv = 1e-4 * np.exp(rng.normal(0.0, 0.5, n))          # strictly positive realised variance
    ret = rng.normal(0.0, 0.01, n)
    vix = 15.0 + 5.0 * rng.random(n)
    return pd.DataFrame(
        {"close": 100.0 + np.cumsum(ret),
         "log_return": ret,
         "vix": vix,
         "realised_variance": rv,
         "rv_daily": np.sqrt(rv),
         "rv_annualised": np.sqrt(rv) * np.sqrt(252)},
        index=idx,
    )


class Persistence(VolatilityModel):
    """Test stub only (NOT a ladder model): forecast next variance = last observed
    variance. Leak-free by construction; used to smoke-test the harness loop."""

    def __init__(self, seed=0):
        super().__init__(name="persistence")

    def fit(self, train):
        pass

    def forecast(self, history):
        return float(history["realised_variance"].iloc[-1])


@pytest.fixture
def persistence_factory():
    return lambda seed=0: Persistence(seed)
