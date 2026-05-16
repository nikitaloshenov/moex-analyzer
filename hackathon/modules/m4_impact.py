"""
M4: Market Impact Model (Linear)
=================================
cost(x) = a(t,s) * x
where x = participation_rate = our_volume / market_volume_1min

По ТЗ: a — матрица (T × N), зависит от времени и инструмента.

Калибровка:
- Если a_fixed задан → константная матрица (для baseline)
- Иначе → Kyle's lambda per ticker (OLS dp ~ pr, no intercept),
  затем rolling по дням для временно́й зависимости
"""

import numpy as np
import pandas as pd
from typing import Optional


class ImpactModel:
    """
    Parameters
    ----------
    a_fixed : float or None
        Если задан — константный коэф. для всех тикеров и времён.
        Если None — калибруется Kyle's lambda из данных.
    a_min, a_max : float
        Границы для откалиброванного a.
    lookback_days : int
        Число дней для rolling-калибровки.
    """

    def __init__(
        self,
        a_fixed: Optional[float] = None,
        a_min:   float = 0.01,
        a_max:   float = 0.05,
        lookback_days: int = 20,
    ):
        self.a_fixed       = a_fixed
        self.a_min         = a_min
        self.a_max         = a_max
        self.lookback_days = lookback_days
        self._calibrated:  pd.DataFrame = pd.DataFrame()   # [seccode, a]  — статика
        self._a_time_wide: pd.DataFrame = pd.DataFrame()   # [date × seccode] — динамика

    # ── Калибровка ───────────────────────────────────────────────────────────
    def calibrate(self, bars_1min: pd.DataFrame) -> "ImpactModel":
        """
        Калибрует a per (date, seccode) через Kyle's lambda.

        Если a_fixed задан — просто заполняет статику константой.
        """
        tickers = bars_1min["seccode"].unique()

        if self.a_fixed is not None:
            self._calibrated = pd.DataFrame({
                "seccode": tickers,
                "a": float(self.a_fixed),
            })
            return self

        # ── Полная калибровка Kyle's lambda ─────────────────────────────
        df = bars_1min.copy()
        df["date"] = df["timestamp"].dt.date

        results = []
        for sec, grp in df.groupby("seccode"):
            grp = grp.sort_values("timestamp")

            # ADV в лотах по дням
            daily_vol = grp.groupby("date")["volume"].sum()
            adv = daily_vol.mean()
            if adv < 1:
                results.append({"seccode": sec, "a": self.a_min})
                continue

            # participation rate proxy: объём бара / (ADV / 51 баров в день)
            grp = grp.copy()
            grp["pr"] = (grp["volume"] / (adv / 51 + 1e-9)).clip(0, 1)
            grp["dp"] = (grp["close"] - grp["open"]).abs() / (grp["mid"].clip(lower=0.01))

            valid = grp.dropna(subset=["pr", "dp"])
            valid = valid[valid["pr"] > 1e-6]

            if len(valid) < 10:
                results.append({"seccode": sec, "a": self.a_min})
                continue

            x = valid["pr"].values
            y = valid["dp"].values
            a_est = np.dot(x, y) / (np.dot(x, x) + 1e-12)
            a_est = float(np.clip(a_est, self.a_min, self.a_max))
            results.append({"seccode": sec, "a": a_est})

        self._calibrated = pd.DataFrame(results)

        # ── Rolling по дням: строим матрицу (date × seccode) ─────────────
        # Для каждого дня пересчитываем OLS на lookback окне
        dates = sorted(df["date"].unique())
        a_rows = []
        for i, d in enumerate(dates):
            lookback_start = dates[max(0, i - self.lookback_days)]
            window = df[(df["date"] >= lookback_start) & (df["date"] <= d)]
            row = {"date": d}
            for sec in tickers:
                grp = window[window["seccode"] == sec]
                if len(grp) < 10 or grp["volume"].sum() < 1:
                    # fallback к статической оценке
                    static = self._calibrated[self._calibrated["seccode"] == sec]["a"]
                    row[sec] = float(static.values[0]) if len(static) else self.a_min
                    continue
                adv_w = grp.groupby("date")["volume"].sum().mean()
                grp = grp.copy()
                grp["pr"] = (grp["volume"] / (adv_w / 51 + 1e-9)).clip(0, 1)
                grp["dp"] = (grp["close"] - grp["open"]).abs() / grp["mid"].clip(lower=0.01)
                v = grp.dropna(subset=["pr", "dp"])
                v = v[v["pr"] > 1e-6]
                if len(v) < 5:
                    static = self._calibrated[self._calibrated["seccode"] == sec]["a"]
                    row[sec] = float(static.values[0]) if len(static) else self.a_min
                    continue
                x, y = v["pr"].values, v["dp"].values
                a_est = float(np.clip(np.dot(x, y) / (np.dot(x, x) + 1e-12),
                                      self.a_min, self.a_max))
                row[sec] = a_est
            a_rows.append(row)

        self._a_time_wide = pd.DataFrame(a_rows).set_index("date")
        return self

    # ── Per-bar impact table ─────────────────────────────────────────────────
    def compute(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Строит таблицу [bar_end_ts, seccode, volume_mkt, a].

        a берётся из rolling-матрицы по дате бара (если она построена),
        иначе из статической калибровки.
        """
        if self._calibrated.empty:
            raise RuntimeError("Call .calibrate(bars_1min) first")

        df = bars_1min[["bar_end_ts", "seccode", "volume", "timestamp"]].copy()
        df = df.rename(columns={"volume": "volume_mkt"})
        df["date"] = df["timestamp"].dt.date

        if not self._a_time_wide.empty:
            # Джойним через дату → seccode из матрицы (T × N)
            a_long = (
                self._a_time_wide
                .reset_index()
                .melt(id_vars="date", var_name="seccode", value_name="a")
            )
            df = df.merge(a_long, on=["date", "seccode"], how="left")
        else:
            # Константная матрица
            df = df.merge(self._calibrated[["seccode", "a"]], on="seccode", how="left")

        df["a"] = df["a"].fillna(self.a_min)
        df["bar_end_ts"] = df["bar_end_ts"].astype("int64")

        return df[["bar_end_ts", "seccode", "volume_mkt", "a"]].reset_index(drop=True)

    def get_a(self, seccode: str) -> float:
        if self._calibrated.empty:
            return self.a_fixed or self.a_min
        row = self._calibrated[self._calibrated["seccode"] == seccode]
        return float(row["a"].values[0]) if len(row) else self.a_min

    def summary(self) -> pd.DataFrame:
        return self._calibrated.sort_values("a") if not self._calibrated.empty else pd.DataFrame()
