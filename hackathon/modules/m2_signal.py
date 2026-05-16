"""
M2: Signal Generation
Input:  bars_5min (long format)
Output: signal_5min — DataFrame[bar_end_ts, seccode, value ∈ [-1,1] | NaN]

Strategy: Cross-sectional momentum with EMA smoothing + order-book imbalance.

signal = clip(z-score(alpha), -1, 1)

alpha = w1 * momentum_ema + w2 * ob_imbalance  (if columns available)
      = momentum_ema                             (otherwise)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Z-score WITHIN each timestamp (cross-sectional normalization).
    """
    out = df.copy()
    grouped = df.groupby("timestamp")[col]
    out["_mean"] = grouped.transform("mean")
    out["_std"]  = grouped.transform("std")
    out[col] = (out[col] - out["_mean"]) / (out["_std"] + 1e-9)
    return out.drop(columns=["_mean", "_std"])


# ─────────────────────────────────────────────
# M2: SIGNAL
# ─────────────────────────────────────────────
class SignalGenerator:
    """
    Computes a cross-sectional alpha signal from 5-min bars.

    Parameters
    ----------
    ema_fast : int
        Fast EMA span (bars) for momentum
    ema_slow : int
        Slow EMA span (bars) for momentum
    delta : float
        Threshold: bars with |signal| < delta → NaN (no trade)
    w_momentum : float
        Weight for momentum component
    w_imbalance : float
        Weight for order-book imbalance component (used if available)
    """

    def __init__(
        self,
        ema_fast: int   = 5,
        ema_slow: int   = 20,
        delta: float    = 0.1,
        w_momentum: float   = 0.7,
        w_imbalance: float  = 0.3,
    ):
        self.ema_fast    = ema_fast
        self.ema_slow    = ema_slow
        self.delta       = delta
        self.w_momentum  = w_momentum
        self.w_imbalance = w_imbalance

    def compute(self, bars_5min: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        bars_5min : DataFrame
            Must have columns: timestamp, seccode, close, [volume, bid, ask, ...]

        Returns
        -------
        signal_5min : DataFrame[bar_end_ts, seccode, value]
        """
        df = bars_5min.copy().sort_values(["seccode", "timestamp"])

        # ── 1. Log returns ─────────────────────────────────────────────────
        df["log_ret"] = (
            df.groupby("seccode")["close"]
            .transform(lambda s: np.log(s).diff())
        )

        # ── 2. Momentum: EMA(fast) - EMA(slow) ─────────────────────────────
        df["ema_fast"] = df.groupby("seccode")["log_ret"].transform(
            lambda s: _ema(s.fillna(0), self.ema_fast)
        )
        df["ema_slow"] = df.groupby("seccode")["log_ret"].transform(
            lambda s: _ema(s.fillna(0), self.ema_slow)
        )
        df["momentum"] = df["ema_fast"] - df["ema_slow"]

        # ── 3. Order-book imbalance (optional) ─────────────────────────────
        has_bid_ask = "bid" in df.columns and "ask" in df.columns
        if has_bid_ask:
            df["ob_imbalance"] = (df["bid"] - df["ask"]) / (df["bid"] + df["ask"] + 1e-9)
            alpha = (
                self.w_momentum  * df["momentum"]
              + self.w_imbalance * df["ob_imbalance"]
            )
        else:
            alpha = df["momentum"]

        df["alpha_raw"] = alpha

        # ── 4. Cross-sectional Z-score ─────────────────────────────────────
        df = _cross_sectional_zscore(df, "alpha_raw")

        # ── 5. Clip to [-1, 1] ─────────────────────────────────────────────
        df["value"] = df["alpha_raw"].clip(-1, 1)

        # ── 6. Apply threshold δ ───────────────────────────────────────────
        df.loc[df["value"].abs() < self.delta, "value"] = np.nan

        # ── 7. Output ──────────────────────────────────────────────────────
        signal = df[["bar_end_ts", "seccode", "value"]].copy()
        signal = signal.dropna(subset=["bar_end_ts"])
        signal["bar_end_ts"] = signal["bar_end_ts"].astype("int64")
        return signal.reset_index(drop=True)
