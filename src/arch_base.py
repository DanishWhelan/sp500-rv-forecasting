"""
Shared base for the arch-library volatility models (GARCH, EGARCH, GJR-GARCH).

Holds the common machinery so each model only declares its arch specification and how to
read its fitted parameters:
  - fit on PERCENT returns (optimiser conditioning) and unscale the forecast variance by
    RETURN_SCALE**2 back to the realised-variance (decimal^2) scale;
  - re-fit by maximum likelihood at each origin (strict walk-forward, refit_every=1);
  - one-step-ahead conditional VARIANCE via arch's own analytic forecast;
  - a param_history record at every refit for parameter validation.

The warm-parameter path (refit_every>1, history longer than the fit sample) is left
unimplemented in the base: the classical models all use strict re-fitting. GARCH(1,1)
overrides it, because the hybrid will later need a one-step GARCH forecast filtered
forward with fixed parameters.
"""
import warnings

import numpy as np
from arch import arch_model

from model_base import VolatilityModel

RETURN_SCALE = 100.0   # fit on percent returns; unscale forecast variance by RETURN_SCALE**2


class ArchVolatilityModel(VolatilityModel):
    """Base class for arch-based conditional-variance models with a constant mean.

    Subclasses set VOL / P / O / Q (the arch specification) and implement _extract() to
    map fitted arch parameters to a record dict including a 'persistence' value.
    """

    VOL = "GARCH"
    P, O, Q = 1, 0, 1

    def __init__(self, name, mean="Constant", dist="normal",
                 target="realised_variance", return_col="log_return"):
        super().__init__(name=name, features=(), target=target, return_col=return_col)
        self.mean = mean
        self.dist = dist
        self._res = None
        self._params = None
        self._fit_len = None
        self.param_history = []   # one dict per refit (parameters + persistence + converged)

    def _scaled_returns(self, frame):
        r = frame[self.return_col].to_numpy(dtype=float)
        return r[~np.isnan(r)] * RETURN_SCALE

    def _extract(self, params):
        """Map an arch params Series to a record dict with keys including 'persistence'.
        Implemented per subclass (parameter names and persistence formula differ)."""
        raise NotImplementedError

    def fit(self, train):
        r = self._scaled_returns(train)
        am = arch_model(r, mean=self.mean, vol=self.VOL, p=self.P, o=self.O, q=self.Q,
                        dist=self.dist, rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # suppress per-fit convergence chatter
            res = am.fit(disp="off", update_freq=0)

        record = self._extract(res.params)
        record["converged"] = bool(getattr(res, "convergence_flag", 0) == 0)
        self._res = res
        self._fit_len = r.size
        self._params = record
        self.param_history.append(record)

    def _forecast_warm(self, r_scaled):
        """One-step variance (percent^2) with fixed params when history extends past the
        fit sample. Only needed for refit_every>1; classical models use strict re-fit."""
        raise NotImplementedError(
            f"{self.name}: warm-parameter forecast (refit_every>1) not implemented; "
            f"the classical models re-fit at every origin.")

    def forecast(self, history):
        if self._params is None:
            raise RuntimeError(f"{self.name}.forecast called before fit")
        r = self._scaled_returns(history)
        if r.size == self._fit_len:
            fc = self._res.forecast(horizon=1, reindex=False)   # arch's own one-step forecast
            h_pct2 = float(np.asarray(fc.variance)[-1, 0])
        else:
            h_pct2 = self._forecast_warm(r)
        return h_pct2 / (RETURN_SCALE ** 2)
