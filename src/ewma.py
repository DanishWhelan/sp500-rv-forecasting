"""
EWMA (RiskMetrics) volatility model -- the naive baseline (rung 1 of the ladder).

The exponentially weighted moving average variance recursion of RiskMetrics (1996):

    sigma^2_{t+1|t} = lambda * sigma^2_{t|t-1} + (1 - lambda) * r_t^2

with the RiskMetrics zero-mean assumption (squared returns, not demeaned). The one-step
-ahead variance forecast is sigma^2_{t+1|t}: the recursion filtered through every return
up to and including the last day of `history`.

lambda is FIXED at the RiskMetrics daily convention 0.94. It is deliberately NOT tuned:
this is the naive benchmark the rest of the ladder must beat, and tuning it would defeat
its purpose (and the integrity of the comparison). It is exposed as a constructor
argument only so the choice is explicit and auditable, not so it gets optimised.

Leakage: forecast() only ever reads returns inside the `history` frame the harness passes
(all index <= origin); the recursion is filtered forward from the start of that frame, so
no future return can enter the forecast. There is no scaling to fit, so fit() is a no-op.
"""
import numpy as np

from model_base import VolatilityModel

RISKMETRICS_LAMBDA = 0.94   # RiskMetrics daily decay factor (Morgan/Reuters, 1996)
VAR_FLOOR = 1e-12           # guards QLIKE (needs a strictly positive variance forecast)


class EWMA(VolatilityModel):
    """RiskMetrics EWMA variance forecaster.

    Parameters
    ----------
    lam : float
        Decay factor lambda in (0, 1). Default 0.94 (RiskMetrics daily). Not tuned.
    name : str, optional
        Identifier for results tables. Defaults to 'ewma'.
    """

    def __init__(self, lam=RISKMETRICS_LAMBDA, name="ewma",
                 target="realised_variance", return_col="log_return"):
        if not 0.0 < lam < 1.0:
            raise ValueError(f"lambda must be in (0, 1); got {lam}")
        # EWMA is a pure-return model: no exogenous features.
        super().__init__(name=name, features=(), target=target, return_col=return_col)
        self.lam = lam

    def fit(self, train):
        # Fixed-parameter model with no scaling to learn; nothing to estimate on train.
        pass

    def forecast(self, history):
        r = history[self.return_col].to_numpy(dtype=float)
        r = r[~np.isnan(r)]                 # first log-return may be NaN before alignment
        if r.size == 0:
            raise ValueError("EWMA.forecast: no returns in history")
        r2 = r ** 2

        # Recursive EWMA, seeded with the first squared return (adjust=False convention).
        # With years of warm-up before the first forecast, the seed is fully washed out
        # by the 0.94 decay, so the forecast depends only on the observed returns.
        sigma2 = r2[0]
        for x in r2[1:]:
            sigma2 = self.lam * sigma2 + (1.0 - self.lam) * x

        return float(max(sigma2, VAR_FLOOR))
