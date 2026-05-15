from __future__ import annotations

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[misc, assignment]

from app.core.config import settings
from app.services.patterns import CandlePatterns


class PatternAnalyzer:
    """
    Модуль детекции сигналов.
    По срезу OHLCV (30-минутные бары) вычисляет паттерны, ATR и уровни TP/SL.
    """

    @staticmethod
    def analyze_all(
        candles,
        ticker="DEFAULT",
        d_sup=None,
        d_res=None,
        ma200_glob=None,
        trades=None
    ) -> dict:
        """
        Полный разбор последней свечи в переданном слайсе данных.
        """
        # --- Вход Polars → pandas (граница с высокопроизводительным движком) ---
        if pl is not None and isinstance(candles, pl.DataFrame):
            candles = candles.to_pandas()

        ps = settings.pattern_signals
        atr_w = ps.atr_window

        # --- Защита: минимум данных для прогрева ATR и скользящих средних ---
        if candles is None or len(candles) < max(30, atr_w + 1):
            return {"signal": "HOLD", "score": 0, "price": 0}

        conf = {"tp_mult": ps.tp_mult, "sl_mult": ps.sl_mult}
        current_price = float(candles["close"].iloc[-1])

        # --- ATR: волатильность для динамических уровней TP/SL ---
        high_low = candles["high"] - candles["low"]
        tr = pd.concat(
            [
                high_low,
                abs(candles["high"] - candles["close"].shift()),
                abs(candles["low"] - candles["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)
        
        atr_series = tr.rolling(atr_w).mean()
        atr = float(atr_series.iloc[-1])
        
        if pd.isna(atr) or atr <= 0:
            atr = current_price * 0.001  # Минимальная заглушка, чтобы не сломать TP/SL

        # --- Объёмная валидация (VSA) ---
        volume_confirmed = CandlePatterns.check_volume(
            candles, multiplier=ps.volume_confirm_multiplier
        )

        # --- Сканирование паттернов Price Action ---
        is_pin = CandlePatterns.is_pin_bar(candles)
        is_engulfing = CandlePatterns.is_engulfing(candles)
        is_railroad = CandlePatterns.is_railroad_tracks(candles)
        any_pattern = is_pin or is_engulfing or is_railroad

        # --- Логика генерации сигналов ---
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

        # --- Вычисление уровней TP/SL от текущей цены рынка ---
        stop_loss = None
        take_profit = None
        
        if "BUY" in res_signal:
            stop_loss = current_price - (atr * conf["sl_mult"])
            take_profit = current_price + (atr * conf["tp_mult"])
        elif "SELL" in res_signal:
            stop_loss = current_price + (atr * conf["sl_mult"])
            take_profit = current_price - (atr * conf["tp_mult"])

        rr_ratio = None
        if stop_loss is not None and take_profit is not None:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk != 0:
                rr_ratio = round(reward / risk, 2)

        return {
            "ticker": str(ticker),
            "score": float(score),
            "signal": str(res_signal),
            "price": float(current_price),
            "stop_loss": float(round(stop_loss, 2)) if stop_loss is not None else None,
            "take_profit": float(round(take_profit, 2)) if take_profit is not None else None,
            "pattern_found": bool(any_pattern),
            "volume_ok": bool(volume_confirmed),
            "atr": float(round(atr, 2)),
            "rr_ratio": float(rr_ratio) if rr_ratio is not None else None,
        }