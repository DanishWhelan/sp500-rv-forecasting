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

Shared arch machinery (scaling, fit, strict re-fit, one-step forecast, param capture)
lives in arch_base.ArchVolatilityModel. This module adds the GARCH(1,1) specification, the
parameter extraction, and the manual one-step recursion used on the warm-parameter path
(refit_every>1) and later by the hybrid.

Future sensitivity check (documented, not a blocker): innovations are Normal here. On the
2000-2022 sample the fitted alpha runs slightly above the canonical S&P 500 value (~0.10 vs
~0.08) with beta correspondingly lower, while persistence (alpha+beta ~ 0.985) is textbook.
Normal innovations and a crisis-heavy sample plausibly explain the elevated alpha. Refitting
with Student-t innovations (dist="t", already supported) is a planned robustness check on the
alpha/beta split; it is not expected to change the persistence or the comparison conclusions.
"""
from arch_base import ArchVolatilityModel, RETURN_SCALE   # noqa: F401  (RETURN_SCALE re-exported)


class GARCH11(ArchVolatilityModel):
    """GARCH(1,1) with a constant mean and (by default) Normal innovations."""

    VOL = "GARCH"
    P, O, Q = 1, 0, 1

    def __init__(self, name="garch", mean="Constant", dist="normal",
                 target="realised_variance", return_col="log_return"):
        super().__init__(name=name, mean=mean, dist=dist, target=target, return_col=return_col)

    def _extract(self, p):
        omega, alpha, beta = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
        mu = float(p["mu"]) if "mu" in p.index else 0.0
        return dict(omega=omega, alpha=alpha, beta=beta, mu=mu, persistence=alpha + beta)

    def _manual_one_step(self, r_scaled):
        """One-step-ahead variance (percent^2) by filtering the GARCH recursion forward
        with the stored parameters. Correct even when `history` extends past the fitted
        sample (the refit_every>1 warm-parameter case used later for the LSTM/hybrid)."""
        p = self._params
        omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
        eps = r_scaled - p["mu"]
        denom = 1.0 - alpha - beta
        sigma2 = omega / denom if denom > 1e-8 else float((eps ** 2).mean())   # unconditional seed
        for e_prev in eps[:-1]:
            sigma2 = omega + alpha * e_prev ** 2 + beta * sigma2               # -> sigma^2_T
        return omega + alpha * eps[-1] ** 2 + beta * sigma2                     # -> sigma^2_{T+1}

    # warm-parameter path (refit_every>1) reuses the manual recursion
    def _forecast_warm(self, r_scaled):
        return self._manual_one_step(r_scaled)
