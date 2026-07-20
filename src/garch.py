"""
GARCH(1,1) volatility model (arch library) -- the core classical baseline (rung 2).

Conditional variance of daily returns:

    r_t   = mu + eps_t
    eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

The one-step-ahead forecast used by the harness is sigma^2_{T+1} = omega + alpha*eps^2_T
+ beta*sigma^2_T, i.e. the forecast for the day AFTER the last row of `history`. This is
exactly the quantity the GARCH-LSTM hybrid will later consume as a feature, so the same
one-step machinery is used here and (later) there.

Re-fit at each origin: with the harness default refit_every=1, fit() re-estimates the
parameters by maximum likelihood on every expanding window, per the strict CLAUDE.md rule.

Scaling: the optimiser is poorly conditioned on raw daily returns (~1e-2), so returns are
fit in PERCENT (x100) and the forecast variance is converted back to decimal^2 (/100^2) so
it is on the same scale as the realised-variance target. alpha and beta are scale-free and
therefore directly comparable to published estimates; omega is in percent^2 units.

Leakage: fit() and forecast() only ever see returns inside `history` (index <= origin).

Future sensitivity check (documented, not a blocker): innovations are Normal here. On the
2000-2022 sample the fitted alpha runs slightly above the canonical S&P 500 value (~0.10 vs
~0.08) with beta correspondingly lower, while persistence (alpha+beta ~ 0.985) is textbook.
Normal innovations and a crisis-heavy sample plausibly explain the elevated alpha. Refitting
with Student-t innovations (dist="t", already supported) is a planned robustness check on the
alpha/beta split; it is not expected to change the persistence or the comparison conclusions.
"""
import warnings

import numpy as np
from arch import arch_model

from model_base import VolatilityModel

RETURN_SCALE = 100.0   # fit on percent returns; unscale forecast variance by RETURN_SCALE**2


class GARCH11(VolatilityModel):
    """GARCH(1,1) with a constant mean and Normal innovations.

    Records one row of fitted parameters per fit() call in `param_history`, so the
    walk-forward parameter path can be validated against the literature.
    """

    def __init__(self, name="garch", mean="Constant", dist="normal",
                 target="realised_variance", return_col="log_return"):
        super().__init__(name=name, features=(), target=target, return_col=return_col)
        self.mean = mean
        self.dist = dist
        self._res = None
        self._params = None
        self._fit_len = None
        self.param_history = []   # one dict per refit: omega, alpha, beta, mu, persistence, converged

    def _scaled_returns(self, frame):
        r = frame[self.return_col].to_numpy(dtype=float)
        return r[~np.isnan(r)] * RETURN_SCALE

    def fit(self, train):
        r = self._scaled_returns(train)
        am = arch_model(r, mean=self.mean, vol="GARCH", p=1, o=0, q=1,
                        dist=self.dist, rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # suppress per-fit convergence chatter
            res = am.fit(disp="off", update_freq=0)

        p = res.params
        omega, alpha, beta = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
        mu = float(p["mu"]) if "mu" in p.index else 0.0
        self._res = res
        self._fit_len = r.size
        self._params = dict(omega=omega, alpha=alpha, beta=beta, mu=mu)

        converged = bool(getattr(res, "convergence_flag", 0) == 0)
        self.param_history.append(dict(omega=omega, alpha=alpha, beta=beta, mu=mu,
                                       persistence=alpha + beta, converged=converged))

    def _manual_one_step(self, r_scaled):
        """One-step-ahead variance (percent^2) by filtering the GARCH recursion forward
        with the stored parameters. Correct even when `history` extends past the fitted
        sample (the refit_every>1 warm-parameter case used later for the LSTM)."""
        p = self._params
        omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
        eps = r_scaled - p["mu"]
        denom = 1.0 - alpha - beta
        sigma2 = omega / denom if denom > 1e-8 else float(np.mean(eps ** 2))  # unconditional seed
        for e_prev in eps[:-1]:
            sigma2 = omega + alpha * e_prev ** 2 + beta * sigma2              # -> sigma^2_T
        return omega + alpha * eps[-1] ** 2 + beta * sigma2                    # -> sigma^2_{T+1}

    def forecast(self, history):
        if self._params is None:
            raise RuntimeError("GARCH11.forecast called before fit")
        r = self._scaled_returns(history)
        if r.size == self._fit_len:
            # strict path (history == fitted sample): use arch's own one-step forecast
            fc = self._res.forecast(horizon=1, reindex=False)
            h_pct2 = float(np.asarray(fc.variance)[-1, 0])
        else:
            # warm path (history longer than fit): filter forward with stored params
            h_pct2 = self._manual_one_step(r)
        return h_pct2 / (RETURN_SCALE ** 2)
