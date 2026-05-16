"""
M4: Market Impact Model (Linear)
=================================
cost(x) = a(t) * x
where x = participation_rate = our_volume / market_volume_1min

Calibration of coefficient `a`:
- Method 1: Fixed (user-provided), default a=0.01
- Method 2: OLS regression on synthetic price impact data
  (in absence of real order data, we estimate from bid-ask spread)
- Method 3: Kyle lambda — regress |price_change| on signed_volume / total_volume

Output: impact_model — DataFrame[bar_end_ts_1min, seccode, volume_mkt, a]
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────
# M4: IMPACT MODEL
# ─────────────────────────────────────────────
class ImpactModel:
    """
    Parameters
    ----------
    a_fixed : float or None
        If provided, use this constant impact coefficient for all tickers.
        If None, calibrate from data.
    a_min, a_max : float
        Bounds for calibrated `a`.
    lookback_days : int
        Number of trading days used for rolling calibration.
    """

    def __init__(
        self,
        a_fixed: Optional[float] = 0.01,
        a_min:  float = 0.01,
        a_max:  float = 0.05,
        lookback_days: int = 20,
    ):
        self.a_fixed       = a_fixed
        self.a_min         = a_min
        self.a_max         = a_max
        self.lookback_days = lookback_days
        self._calibrated: pd.DataFrame = pd.DataFrame()

    # ── Calibration ─────────────────────────────────────────────────────────
    def calibrate(self, bars_1min: pd.DataFrame) -> "ImpactModel":
        """
        Estimate `a` per ticker using Kyle's lambda approach.

        Kyle's lambda: regress |Δprice| ~ participation_rate
        We use synthetic participation_rate = volume_bar / ADV_daily

        If no bid/ask available, fall back to spread-based estimate:
            a ≈ half_spread / mid

        Populates self._calibrated: DataFrame[seccode, a]
        """
        if self.a_fixed is not None:
            tickers = bars_1min["seccode"].unique()
            self._calibrated = pd.DataFrame({
                "seccode": tickers,
                "a": self.a_fixed,
            })
            return self

        results = []
        for sec, grp in bars_1min.groupby("seccode"):
            grp = grp.sort_values("timestamp").copy()

            # Daily ADV
            adv = (
                grp.groupby(grp["timestamp"].dt.date)["volume"].sum().mean()
            )
            if adv == 0:
                results.append({"seccode": sec, "a": 0.01})
                continue

            # Participation rate proxy
            grp["pr"] = grp["volume"] / (adv / 51 + 1e-9)  # ~51 bars per day
            grp["pr"] = grp["pr"].clip(0, 1)

            # Price change (relative)
            grp["dp"] = (grp["close"] - grp["open"]).abs() / (grp["mid"] + 1e-9)

            # OLS: dp = a * pr  (no intercept, impact model)
            valid = grp.dropna(subset=["pr", "dp"])
            valid = valid[valid["pr"] > 0]

            if len(valid) < 10:
                results.append({"seccode": sec, "a": 0.01})
                continue

            x = valid["pr"].values
            y = valid["dp"].values
            # OLS no-intercept: a = Σ(x*y) / Σ(x²)
            a_est = np.dot(x, y) / (np.dot(x, x) + 1e-12)
            a_est = float(np.clip(a_est, self.a_min, self.a_max))
            results.append({"seccode": sec, "a": a_est})

        self._calibrated = pd.DataFrame(results)
        return self

    # ── Per-bar impact table ─────────────────────────────────────────────────
    def compute(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Build the per-1-min-bar impact table.

        Returns
        -------
        impact_model : DataFrame[bar_end_ts, seccode, volume_mkt, a]
        """
        if self._calibrated.empty:
            raise RuntimeError("Call .calibrate(bars_1min) first")

        df = bars_1min[["bar_end_ts", "seccode", "volume"]].copy()
        df = df.rename(columns={"volume": "volume_mkt"})

        # Merge calibrated `a` per ticker
        df = df.merge(self._calibrated[["seccode", "a"]], on="seccode", how="left")
        df["a"] = df["a"].fillna(0.01)

        df["bar_end_ts"] = df["bar_end_ts"].astype("int64")
        return df.reset_index(drop=True)

    def get_a(self, seccode: str) -> float:
        """Return calibrated impact coefficient for a ticker."""
        if self._calibrated.empty:
            return self.a_fixed or 0.01
        row = self._calibrated[self._calibrated["seccode"] == seccode]
        return float(row["a"].values[0]) if len(row) else 0.01

    def summary(self) -> pd.DataFrame:
        return self._calibrated.sort_values("a")
