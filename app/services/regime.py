"""
Режим рынка: флэт vs тренд по расстоянию быстрой и медленной MA.

Используется в бэктесте: во флэте можно запретить слабые сигналы (см. settings.regime).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from app.core.config import settings

RegimeLabel = Literal["uptrend", "downtrend", "flat", "unknown"]


# -----------------------------------------------------------------------------
# Классификация
# -----------------------------------------------------------------------------


def classify_regime(candles: pl.DataFrame) -> RegimeLabel:
    """По последним барам: uptrend | downtrend | flat | unknown."""
    rc = settings.regime
    need = rc.ma_slow + 5
    if candles.height < need:
        return "unknown"

    row = (
        candles.select(
            [
                pl.col("close")
                .rolling_mean(window_size=rc.ma_fast, min_periods=rc.ma_fast)
                .alias("_maf"),
                pl.col("close")
                .rolling_mean(window_size=rc.ma_slow, min_periods=rc.ma_slow)
                .alias("_mas"),
                pl.col("close").alias("_c"),
            ]
        )
        .drop_nulls()
        .tail(1)
    )
    if row.is_empty():
        return "unknown"
    maf = float(row["_maf"][0])
    mas = float(row["_mas"][0])
    c = float(row["_c"][0])
    if c <= 0 or not all(map(lambda x: x == x, (maf, mas, c))):  # NaN check
        return "unknown"

    rel = abs(maf - mas) / c
    if rel < rc.flat_band:
        return "flat"
    if maf > mas:
        return "uptrend"
    return "downtrend"


def regime_allows_signal(signal: str, regime: RegimeLabel) -> bool:
    """Разрешить вход по сигналу с учётом режима (настройки в settings.regime)."""
    if not settings.regime.enabled or regime == "unknown":
        return True
    if regime == "flat":
        if settings.regime.allow_strong_in_flat:
            return "STRONG" in signal
        return False
    return True
