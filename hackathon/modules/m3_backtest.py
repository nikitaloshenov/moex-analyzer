"""
M3: Baseline Backtest (zero-impact, execution at mid)
Input:  signal_5min, bars_5min
Output: backtest_baseline — DataFrame per bar with pos, pnl_mid, cum_pnl_mid

Logic:
  pos_t  = alpha_{t-1}   (signal generated at end of bar t-1 → position at t)
  pnl_t  = pos_t * (close_t / open_t - 1)   ≈ pos * mid_return
"""

import numpy as np
import pandas as pd


class BaselineBacktest:
    """
    Parameters
    ----------
    delta : float
        Signal threshold — bars with |signal| < delta are treated as flat (pos=0).
        Should match the threshold used in M2.
    """

    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def run(
        self,
        signal_5min: pd.DataFrame,
        bars_5min:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        signal_5min : DataFrame[bar_end_ts, seccode, value]
        bars_5min   : DataFrame[timestamp, seccode, open, close, mid, bar_end_ts]

        Returns
        -------
        backtest_baseline : DataFrame[bar_end_ts, seccode, pos, pnl_mid, cum_pnl_mid]
        """
        # ── Align signal → next bar position ──────────────────────────────
        # signal at bar_end_ts T → position entering bar that STARTS at T
        sig = signal_5min.copy()
        sig["value"] = sig["value"].fillna(0)
        sig.loc[sig["value"].abs() < self.delta, "value"] = 0.0

        # bars_5min: we need (open, close) per bar, indexed by bar_end_ts
        bars = bars_5min[["timestamp", "seccode", "open", "close", "mid", "bar_end_ts"]].copy()
        bars = bars.sort_values(["seccode", "timestamp"])

        # For each ticker, shift signal forward by 1 bar to get position
        result_parts = []
        for sec, grp in bars.groupby("seccode"):
            grp = grp.sort_values("timestamp").copy()

            # Get corresponding signals
            sig_sec = sig[sig["seccode"] == sec].sort_values("bar_end_ts")

            # Map signal bar_end_ts → position entering NEXT bar
            # bar_end_ts of bar T matches timestamp of bar T+1 (start)
            # We merge on: bar_end_ts of signal == bar_end_ts of PREVIOUS bar
            # Equivalently: signal bar_end_ts at T → pos at next row
            grp = grp.merge(
                sig_sec[["bar_end_ts", "value"]].rename(columns={"value": "alpha_prev"}),
                on="bar_end_ts",
                how="left"
            )
            # Shift: position at bar t = alpha from bar t-1
            grp["pos"] = grp["alpha_prev"].shift(1).fillna(0)

            # PnL: pos * intra-bar return (open→close)
            grp["ret_bar"] = grp["close"] / grp["open"] - 1
            grp["pnl_mid"] = grp["pos"] * grp["ret_bar"]
            grp["cum_pnl_mid"] = grp["pnl_mid"].cumsum()

            result_parts.append(grp[["bar_end_ts", "seccode", "pos", "pnl_mid", "cum_pnl_mid"]])

        backtest = pd.concat(result_parts, ignore_index=True)
        backtest["bar_end_ts"] = backtest["bar_end_ts"].astype("int64")
        return backtest

    def summary(self, backtest: pd.DataFrame) -> dict:
        """Portfolio-level statistics."""
        port = backtest.groupby("bar_end_ts")["pnl_mid"].sum().reset_index()
        port["cum_pnl"] = port["pnl_mid"].cumsum()

        pnl = port["pnl_mid"]
        sharpe = (pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(252 * 51)  # ~51 bars/day

        hit_rate = (backtest["pnl_mid"] > 0).sum() / max((backtest["pos"] != 0).sum(), 1)

        return {
            "sharpe_ratio":     round(sharpe, 4),
            "hit_rate":         round(hit_rate, 4),
            "total_pnl_mid":    round(port["cum_pnl"].iloc[-1], 6) if len(port) else 0,
            "mean_pnl_per_bar": round(pnl.mean(), 8),
            "std_pnl_per_bar":  round(pnl.std(), 8),
            "n_bars":           len(port),
        }
