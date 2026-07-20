"""
HAR-RV: Heterogeneous AutoRegressive model of Realised Volatility (Corsi 2009)
-- the strong realised-volatility benchmark (final classical rung).

Structurally different from the GARCH family: it does not use returns at all. It regresses
next-day realised VOLATILITY on three backward-looking averages of its own past realised
volatility -- daily (1 day), weekly (5 days) and monthly (22 days):

    v_{t+1} = b0 + b_d v^{(d)}_t + b_w v^{(w)}_t + b_m v^{(m)}_t + e_{t+1}

    v^{(d)}_t = v_t
    v^{(w)}_t = mean(v_{t-4} .. v_t)          (5 trading days, ending at t)
    v^{(m)}_t = mean(v_{t-21} .. v_t)         (22 trading days, ending at t)

where v_t = sqrt(realised_variance_t) is daily realised volatility. The forecast is squared
back to a variance for scoring (QLIKE/RMSE), so HAR plugs into the same harness as the rest.

LEAKAGE BOUNDARY (the whole point of getting this model right)
-------------------------------------------------------------
Every predictor for the target day t+1 is built from realised volatility on days <= t. The
three components all END at day t (the origin); none of them touches v_{t+1}. In the training
design the alignment is an explicit FORWARD shift: the row of predictors dated t is paired
with target v_{t+1} (`vol.shift(-1)`), so the target day's own RV is never among its own
predictors. See _design(), and the leak-boundary test in tests/test_har.py which asserts that
for a strictly increasing series every predictor is strictly less than the target it predicts
(impossible if any predictor had peeked at the larger target day).
"""
import numpy as np
import pandas as pd

from model_base import VolatilityModel

WEEKLY, MONTHLY = 5, 22    # HAR component windows (trading days), Corsi (2009)
VOL_FLOOR = 1e-6           # floor the volatility forecast so the squared variance stays > 0


class HARRV(VolatilityModel):
    """HAR-RV regression on realised volatility (OLS, re-fit at each origin)."""

    def __init__(self, name="har", target="realised_variance", weekly=WEEKLY, monthly=MONTHLY):
        # HAR uses only lagged realised volatility (the target's own history); no features.
        super().__init__(name=name, features=(), target=target)
        self.weekly = weekly
        self.monthly = monthly
        self._coef = None            # [b0, b_d, b_w, b_m]
        self.coef_history = []       # one row of coefficients per refit (for the record)

    def _components(self, vol):
        """Daily / weekly / monthly backward averages of realised volatility.

        Each component at index s uses only v on days <= s (rolling looks backward), so a
        component value dated s never contains information from day s+1 onward.
        """
        daily = vol
        weekly = vol.rolling(self.weekly).mean()
        monthly = vol.rolling(self.monthly).mean()
        return daily, weekly, monthly

    def _design(self, train):
        """Build the training design: predictors dated s, target = v_{s+1}.

        The `shift(-1)` on the target is the leak boundary: predictors dated s (which end at
        v_s) are matched with the NEXT day's volatility. Rows with any NaN (the leading
        rolling window, and the final row that has no next-day target) are dropped.
        """
        vol = np.sqrt(train[self.target].astype(float))
        daily, weekly, monthly = self._components(vol)
        design = pd.DataFrame({"daily": daily, "weekly": weekly, "monthly": monthly,
                               "target": vol.shift(-1)}).dropna()
        return design

    def fit(self, train):
        d = self._design(train)
        A = np.column_stack([np.ones(len(d)), d["daily"].to_numpy(),
                             d["weekly"].to_numpy(), d["monthly"].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, d["target"].to_numpy(), rcond=None)
        self._coef = coef
        self.coef_history.append(dict(b0=coef[0], b_daily=coef[1],
                                      b_weekly=coef[2], b_monthly=coef[3]))

    def forecast(self, history):
        if self._coef is None:
            raise RuntimeError("HARRV.forecast called before fit")
        vol = np.sqrt(history[self.target].astype(float))
        # Predictors for day t+1, all ending at the origin day t (history's last row).
        daily = float(vol.iloc[-1])
        weekly = float(vol.iloc[-self.weekly:].mean())
        monthly = float(vol.iloc[-self.monthly:].mean())
        b0, b_d, b_w, b_m = self._coef
        vol_hat = b0 + b_d * daily + b_w * weekly + b_m * monthly
        vol_hat = max(vol_hat, VOL_FLOOR)
        return vol_hat ** 2          # square the volatility forecast back to a variance
