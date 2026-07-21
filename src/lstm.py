"""
LSTM volatility model (rung 5, the deep model).

Forecasts one-step-ahead conditional VARIANCE by predicting standardised LOG realised
variance with an LSTM over a fixed lookback window, then inverting the transform:

    variance_hat = exp( unstandardise( net(window) ) )

so the output is a positive variance by construction (no flooring needed).

Design (fixed a priori; see the design note in the project history):
  - target: log(realised_variance), standardised on the training window only
  - inputs: per-timestep features (default log realised variance + VIX); ablation drops a
    feature by CONSTRUCTING the model with a shorter `features` tuple (no code change)
  - lookback L = 22 trading days (matches HAR's monthly horizon, for a fair comparison)
  - 1-layer LSTM, hidden 32, Linear head, MSE loss on standardised log-RV, Adam
  - early stopping on the chronological TAIL of the training window (still < origin)

LEAKAGE-SENSITIVE POINTS (all enforced here, tested in tests/test_lstm.py):
  1. Windowing: the input window for day t+1 ends at day t; the target day is never inside
     its own window. Training pairs a window ending at s with target log-RV_{s+1}.
  2. Scaling fit on TRAIN ONLY: feature/target means and stds are computed on the training
     window passed to fit() and stored; forecast() applies those stored constants. Under
     refit_every>1, the scaler and weights come from the last refit origin r (<= the current
     origin), so applying them to newer input rows uses only information available by r --
     leak-free.
  3. Seeding: torch/numpy seeded per fit for reproducibility; report mean +/- std over seeds.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from model_base import VolatilityModel

LOOKBACK = 22          # trading-day window (matches HAR monthly horizon)
HIDDEN = 32
MAX_EPOCHS = 100
PATIENCE = 10          # early-stopping patience on validation MSE
LR = 1e-3
BATCH = 64
VAL_FRAC = 0.15        # chronological tail of train used for early stopping
LOGVAR_CLIP = (-30.0, 5.0)   # clip predicted log-variance before exp (numerical safety)
STD_EPS = 1e-8


class _LSTMNet(nn.Module):
    """1-layer LSTM regressor: sequence -> last hidden state -> scalar (standardised log-RV)."""

    def __init__(self, n_features, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                     # x: [B, L, F]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # [B]


class LSTMModel(VolatilityModel):
    """LSTM forecaster of one-step-ahead conditional variance."""

    def __init__(self, seed=0, features=("realised_variance", "vix"),
                 target="realised_variance", lookback=LOOKBACK, hidden=HIDDEN,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, batch=BATCH,
                 val_frac=VAL_FRAC, name="lstm"):
        super().__init__(name=name, features=features, target=target)
        self.seed = seed
        self.lookback = lookback
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.batch = batch
        self.val_frac = val_frac
        self._net = None
        self._scaler = None      # dict: feat_mean, feat_std, tgt_mean, tgt_std

    # ---- feature / target construction (log-transform the realised-variance column) ----
    def _raw_features(self, frame):
        cols = []
        for c in self.features:
            v = frame[c].to_numpy(dtype=float)
            if c == self.target:
                v = np.log(v)          # realised variance enters in logs
            cols.append(v)
        return np.column_stack(cols)   # [N, F]

    def _raw_target(self, frame):
        return np.log(frame[self.target].to_numpy(dtype=float))   # [N] log realised variance

    @staticmethod
    def _make_sequences(feats, target, lookback):
        """Windows ending at s paired with target at s+1 (the forward-shift leak boundary).

        Returns X [n, L, F], y [n], ends [n] where ends[i] = s is the last day IN the window
        (strictly before the target day s+1). No window ever contains its own target day.
        """
        n = len(feats)
        X, y, ends = [], [], []
        for s in range(lookback - 1, n - 1):        # need s-L+1 >= 0 and target s+1 <= n-1
            X.append(feats[s - lookback + 1:s + 1])
            y.append(target[s + 1])
            ends.append(s)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(ends)

    def _fit_scaler(self, feats, target):
        self._scaler = dict(
            feat_mean=feats.mean(axis=0), feat_std=feats.std(axis=0) + STD_EPS,
            tgt_mean=float(target.mean()), tgt_std=float(target.std()) + STD_EPS)

    def _standardise_feats(self, feats):
        s = self._scaler
        return (feats - s["feat_mean"]) / s["feat_std"]

    def fit(self, train):
        # seed everything for reproducibility (fresh init + deterministic batching per fit)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        feats = self._raw_features(train)
        target = self._raw_target(train)
        # scaler fit on TRAIN ONLY, then applied to the training features and target
        self._fit_scaler(feats, target)
        z_feats = self._standardise_feats(feats)
        z_target = (target - self._scaler["tgt_mean"]) / self._scaler["tgt_std"]

        X, y, _ = self._make_sequences(z_feats, z_target, self.lookback)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        # chronological train/validation split (validation = most recent tail of train)
        n = len(X)
        n_val = max(1, int(n * self.val_frac))
        X_tr, y_tr = X[:n - n_val], y[:n - n_val]
        X_val, y_val = X[n - n_val:], y[n - n_val:]

        net = _LSTMNet(X.shape[2], self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val, best_state, wait = float("inf"), None, 0
        n_tr = len(X_tr)
        for _ in range(self.max_epochs):
            net.train()
            for batch_idx in np.array_split(rng.permutation(n_tr),
                                            max(1, n_tr // self.batch)):
                opt.zero_grad()
                loss = loss_fn(net(X_tr[batch_idx]), y_tr[batch_idx])
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(X_val), y_val))
            if val_loss < best_val - 1e-6:
                best_val, best_state, wait = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net

    def forecast(self, history):
        if self._net is None:
            raise RuntimeError("LSTMModel.forecast called before fit")
        feats = self._raw_features(history)
        z = self._standardise_feats(feats)
        window = z[-self.lookback:]                       # last L rows, ending at the origin day
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)   # [1, L, F]
        with torch.no_grad():
            pred_std = float(self._net(x).item())
        log_var = pred_std * self._scaler["tgt_std"] + self._scaler["tgt_mean"]
        log_var = min(max(log_var, LOGVAR_CLIP[0]), LOGVAR_CLIP[1])
        return float(np.exp(log_var))                     # positive variance by construction
