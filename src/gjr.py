"""
GJR-GARCH(1,1) volatility model (Glosten, Jagannathan, Runkle 1993) -- asymmetric rung.

Conditional variance with a leverage term:

    sigma^2_t = omega + alpha eps^2_{t-1} + gamma eps^2_{t-1} I(eps_{t-1} < 0)
                      + beta sigma^2_{t-1}

The asymmetry coefficient gamma is added only after negative shocks: a POSITIVE gamma means
negative returns raise next-period volatility more than positive returns of the same size
(the equity leverage effect). Under a symmetric innovation distribution the stationarity /
persistence measure is alpha + beta + gamma/2 (the 1/2 is the probability of a negative shock).

In the arch library GJR-GARCH is the standard GARCH volatility process with the asymmetry
order o=1. Shared machinery lives in arch_base.ArchVolatilityModel.
"""
from arch_base import ArchVolatilityModel


class GJRGARCH11(ArchVolatilityModel):
    """GJR-GARCH(1,1) with a constant mean and (by default) Normal innovations."""

    VOL = "GARCH"
    P, O, Q = 1, 1, 1

    def __init__(self, name="gjr", mean="Constant", dist="normal",
                 target="realised_variance", return_col="log_return"):
        super().__init__(name=name, mean=mean, dist=dist, target=target, return_col=return_col)

    def _extract(self, p):
        omega = float(p["omega"])
        alpha = float(p["alpha[1]"])
        gamma = float(p["gamma[1]"])          # asymmetry: gamma > 0 => leverage effect
        beta = float(p["beta[1]"])
        mu = float(p["mu"]) if "mu" in p.index else 0.0
        # symmetric-innovation persistence measure for GJR
        return dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta, mu=mu,
                    persistence=alpha + beta + 0.5 * gamma)
