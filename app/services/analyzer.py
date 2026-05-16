from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from app.core.config import settings


class PatternAnalyzer:
    """
    M2: clean 5-minute signal generator.

    Output contract:
        signal_5min:
            bar_end_ts: int64
            seccode: str
            value: float64 in [-1, 1] or NaN
    """

    @staticmethod
    def _ts_int(value) -> int:
        if hasattr(value, "timestamp"):
            return int(value.timestamp() * 1_000_000_000)
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def build_signal_5min(bars_5m: pl.DataFrame, ticker: str = "DEFAULT") -> pl.DataFrame:
        if bars_5m is None or bars_5m.is_empty():
            return pl.DataFrame(
                schema={
                    "bar_end_ts": pl.Int64,
                    "seccode": pl.Utf8,
                    "value": pl.Float64,
                }
            )

        ps = settings.pattern_signals
        df = bars_5m.sort("begin").to_pandas()
        atr_window = int(getattr(ps, "atr_window", 20))

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df.get("volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

        ema_fast = close.ewm(span=10, adjust=False, min_periods=10).mean()
        ema_slow = close.ewm(span=30, adjust=False, min_periods=30).mean()
        ema_slope = ema_fast - ema_fast.shift(3)
        rsi = PatternAnalyzer._rsi(close, atr_window)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / atr_window, adjust=False, min_periods=atr_window).mean()
        safe_atr = atr.replace(0.0, np.nan)

        prev_high = high.shift(1).rolling(20, min_periods=20).max()
        prev_low = low.shift(1).rolling(20, min_periods=20).min()
        avg_vol = volume.shift(1).rolling(20, min_periods=20).mean()

        buy_size = pd.to_numeric(df.get("buy_size", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        sell_size = pd.to_numeric(df.get("sell_size", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        liquidity = (buy_size + sell_size).replace(0.0, np.nan)
        imbalance = (buy_size - sell_size) / liquidity

        value = pd.Series(np.nan, index=df.index, dtype="float64")

        trend_up = (ema_fast > ema_slow) & (ema_slope > 0)
        trend_down = (ema_fast < ema_slow) & (ema_slope < 0)

        ema_distance_atr = (close - ema_fast).abs() / safe_atr
        volume_confirmed = (volume > avg_vol * 1.05) & (volume < avg_vol * 3.0)
        volatility_stable = true_range < safe_atr * 2.0
        not_overextended = ema_distance_atr < 2.5

        breakout_long = (
            (close > prev_high)
            & volume_confirmed
            & volatility_stable
            & not_overextended
            & (imbalance.fillna(0.0) >= 0.0)
        )
        breakout_short = (
            (close < prev_low)
            & volume_confirmed
            & volatility_stable
            & not_overextended
            & (imbalance.fillna(0.0) <= 0.0)
        )

        # Simple defensible alpha:
        # trend-following breakout + quality filters to avoid chasing spikes.
        value.loc[trend_up & breakout_long & (rsi < 68)] = 1.0
        value.loc[trend_down & breakout_short & (rsi > 32)] = -1.0

        cooldown_bars = 2
        last_signal_idx = -10**9
        for idx, raw_value in enumerate(value.to_numpy()):
            if not np.isfinite(raw_value):
                continue
            if idx - last_signal_idx <= cooldown_bars:
                value.iat[idx] = np.nan
                continue
            last_signal_idx = idx

        out = pl.DataFrame(
            {
                "bar_end_ts": [PatternAnalyzer._ts_int(x) for x in df["begin"]],
                "seccode": [ticker] * len(df),
                "value": value.tolist(),
            }
        )
        return out

    @staticmethod
    def analyze_all(
        candles,
        ticker: str = "DEFAULT",
        d_sup=None,
        d_res=None,
        ma200_glob=None,
        trades=None,
    ) -> dict:
        """
        Legacy-compatible single-bar signal output.
        Internally uses build_signal_5min and returns BUY/SELL/HOLD.
        """
        if isinstance(candles, pl.DataFrame):
            bars = candles
        else:
            bars = pl.from_pandas(candles)

        signal_df = PatternAnalyzer.build_signal_5min(bars, ticker=ticker)
        if signal_df.is_empty():
            return {"ticker": ticker, "signal": "HOLD", "score": 0.0, "price": 0.0}

        val = signal_df["value"][-1]
        price = float(bars["close"][-1]) if "close" in bars.columns and bars.height else 0.0

        if val is None or not np.isfinite(float(val)) or abs(float(val)) < 1e-12:
            signal = "HOLD"
            score = 0.0
        elif float(val) > 0:
            signal = "BUY"
            score = 60.0
        else:
            signal = "SELL"
            score = -60.0

        return {
            "ticker": ticker,
            "signal": signal,
            "score": score,
            "price": price,
            "pattern_found": False,
            "volume_ok": True,
            "atr": 0.0,
            "rr_ratio": None,
        }
