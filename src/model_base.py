"""
Model interface for the volatility-forecasting comparison.

Every model in the ladder (EWMA, GARCH-family, HAR-RV, LSTM, GARCH-LSTM hybrid)
implements this ONE interface, so all of them plug into the same walk-forward
evaluation harness (src/harness.py) and are scored identically.

Design contract
---------------
- Models forecast a one-step-ahead conditional VARIANCE (same scale as the
  realised-variance proxy). The harness owns all slicing, scoring and significance
  testing; a model only estimates parameters and produces a single forecast.
- fit() and forecast() receive ONLY historical data (every row has index <= origin).
  They must never reach beyond the frame they are handed. This is the structural
  guarantee that keeps the study leak-free: a model cannot use information it is
  never given.
- Feature ablation is first-class. A model is configured with an explicit list of
  input feature columns via `features`. Removing the VIX feature (or the GARCH
  feature in the hybrid) is done by CONSTRUCTING the model with a smaller `features`
  list, never by editing model code. This is what the ablation study needs.
"""
from abc import ABC, abstractmethod

import pandas as pd


class VolatilityModel(ABC):
    """Abstract base for all volatility models.

    Parameters
    ----------
    name : str
        Short identifier used in results tables and output filenames.
    features : sequence of str
        Input columns the model is ALLOWED to consume from the modelling frame.
        Ablation runs pass a reduced list (e.g. drop 'vix'); the model code does not
        change. May be empty for pure-return models (e.g. a GARCH on returns uses
        `return_col`, not exogenous features).
    target : str
        Column holding the realised-variance proxy the model predicts and is scored on.
    return_col : str
        Column holding log returns (used by the GARCH-family models).
    """

    def __init__(self, name, features=(), target="realised_variance",
                 return_col="log_return"):
        self.name = name
        self.features = tuple(features)
        self.target = target
        self.return_col = return_col

    def _select_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """Return only the allowed feature columns from `history`.

        Enforces the ablation contract: a model can never accidentally consume a
        feature outside `self.features`. Raises if a configured feature is absent so
        that an ablation typo fails loudly rather than silently using the wrong inputs.
        """
        missing = [c for c in self.features if c not in history.columns]
        if missing:
            raise KeyError(f"{self.name}: configured feature(s) not in data: {missing}")
        return history.loc[:, list(self.features)]

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Estimate parameters using ONLY rows in `train` (all index <= origin).

        Fit ALL scaling/normalisation here, on train only, and store it on self.
        Called by the harness every `refit_every` origins.
        """

    @abstractmethod
    def forecast(self, history: pd.DataFrame) -> float:
        """One-step-ahead conditional VARIANCE forecast.

        Predicts the day immediately AFTER `history`'s last row, using the parameters
        from the most recent fit() and the observations in `history` (all index <=
        origin). `history` is the full expanding window from the start of the sample.

        Must return a strictly positive, finite VARIANCE (not a volatility, not a log).
        The harness validates this every step and raises if violated, so there is no
        need to score an ill-defined forecast; still, prefer flooring degenerate cases
        in the model to returning a value that would trip the guard.
        """
