"""
EGARCH(1,1) volatility model (Nelson 1991) -- asymmetric rung (arch library).

Log conditional variance:

    ln sigma^2_t = omega + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1} + beta ln sigma^2_{t-1}

with z_{t-1} = eps_{t-1} / sigma_{t-1}. The asymmetry (leverage) term is gamma: a NEGATIVE
gamma means negative return shocks raise next-period volatility more than positive shocks of
the same size, the empirically dominant pattern for equity indices. Because the model is
specified in logs, sigma^2 is positive by construction with no parameter constraints, and
persistence is simply the log-variance AR coefficient beta.

Shared arch machinery lives in arch_base.ArchVolatilityModel; this module only sets the
EGARCH specification and reads its parameters.
"""
from arch_base import ArchVolatilityModel


class EGARCH11(ArchVolatilityModel):
    """EGARCH(1,1) with a constant mean and (by default) Normal innovations."""

    VOL = "EGARCH"
    P, O, Q = 1, 1, 1

    def __init__(self, name="egarch", mean="Constant", dist="normal",
                 target="realised_variance", return_col="log_return"):
        super().__init__(name=name, mean=mean, dist=dist, target=target, return_col=return_col)

    def _extract(self, p):
        omega = float(p["omega"])
        alpha = float(p["alpha[1]"])
        gamma = float(p["gamma[1]"])          # asymmetry: gamma < 0 => leverage effect
        beta = float(p["beta[1]"])
        mu = float(p["mu"]) if "mu" in p.index else 0.0
        # EGARCH is an AR(1) in log-variance; persistence is |beta| (stationary if < 1).
        return dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta, mu=mu, persistence=beta)
