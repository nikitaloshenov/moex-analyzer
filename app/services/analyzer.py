"""
Модуль детекции сигналов (V2): PatternAnalyzer + CandlePatterns.

Роль: по срезу OHLCV (pandas или Polars) вернуть сигнал, score, TP/SL по ATR.
Polars здесь только на входе — внутри для паттернов используется pandas (CandlePatterns).
Бэктест на Polars и ликвидность: app.services.backtest.
"""

from __future__ import annotations

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[misc, assignment]

from app.core.config import settings
from app.services.patterns import CandlePatterns


# -----------------------------------------------------------------------------
# Класс анализатора
# -----------------------------------------------------------------------------


class PatternAnalyzer:
    """Пороги ATR / объёма читаются из app.core.config.settings.pattern_signals."""

    @staticmethod
    def analyze_all(
    candles,
    ticker="DEFAULT",
    d_sup=None,
    d_res=None,
    ma200_glob=None,
    trades=None
    ):
        """
        Полный разбор последней свечи в серии.

        candles: pandas.DataFrame или polars.DataFrame (см. backtest: слайс df.slice(0, i+1)).
        ma200_glob: зарезервировано под будущую фильтрацию по MA200.
        """
        # --- Вход Polars → pandas (граница с модулем backtest) ---
        if pl is not None and isinstance(candles, pl.DataFrame):
            candles = candles.to_pandas()

        # --- Минимум данных ---
        if candles is None or len(candles) < 30:
            return {"signal": "HOLD", "score": 0, "price": 0}

        ps = settings.pattern_signals
        conf = {"tp_mult": ps.tp_mult, "sl_mult": ps.sl_mult}
        atr_w = ps.atr_window
        current_price = float(candles["close"].iloc[-1])

        # --- ATR: волатильность для TP/SL ---
        high_low = candles["high"] - candles["low"]
        tr = pd.concat(
            [
                high_low,
                abs(candles["high"] - candles["close"].shift()),
                abs(candles["low"] - candles["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_w).mean().iloc[-1]

        # --- Объём (VSA) ---
        volume_confirmed = CandlePatterns.check_volume(
            candles, multiplier=ps.volume_confirm_multiplier
        )

        # --- Паттерны price action ---
        is_pin = CandlePatterns.is_pin_bar(candles)
        is_engulfing = CandlePatterns.is_engulfing(candles)
        is_railroad = CandlePatterns.is_railroad_tracks(candles)
        any_pattern = is_pin or is_engulfing or is_railroad

        # --- Сигнал: паттерн + объём ---
        direction_up = candles["close"].iloc[-1] > candles["open"].iloc[-1]
        res_signal = "HOLD"
        score = 0

        if volume_confirmed and any_pattern:
            if direction_up:
                res_signal = "STRONG BUY" if is_engulfing else "BUY"
                score = 80 if is_engulfing else 60
            else:
                res_signal = "STRONG SELL" if is_engulfing else "SELL"
                score = -80 if is_engulfing else -60

        # --- Уровни TP/SL от текущей цены и ATR ---
        stop_loss = None
        take_profit = None
        if "BUY" in res_signal:
            stop_loss = current_price - (atr * conf["sl_mult"])
            take_profit = current_price + (atr * conf["tp_mult"])
        elif "SELL" in res_signal:
            stop_loss = current_price + (atr * conf["sl_mult"])
            take_profit = current_price - (atr * conf["tp_mult"])

        rr_ratio = None

        if stop_loss and take_profit:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk != 0:
                rr_ratio = round(reward / risk, 2)
        return {
            "ticker": str(ticker),
            "score": float(score),
            "signal": str(res_signal),
            "price": float(current_price),
            "stop_loss": (
                float(round(stop_loss, 2))
                if stop_loss is not None else None
            ),
            "take_profit": (
                float(round(take_profit, 2))
                if take_profit is not None else None
            ),
            "pattern_found": bool(any_pattern),
            "volume_ok": bool(volume_confirmed),
            "atr": float(round(atr, 2)),
            "rr_ratio": (
                float(rr_ratio)
                if rr_ratio is not None else None
            ),
        }