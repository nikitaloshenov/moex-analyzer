"""
Data ingestion: сырые тики / orderlog → OHLCV на Polars.

Роль: агрегация по временному интервалу (resample). Дальше тот же df можно
отдавать в Moex-подобный контракт (begin, open, high, low, close, volume).
"""

from __future__ import annotations

import polars as pl


# -----------------------------------------------------------------------------
# Тики → свечи
# -----------------------------------------------------------------------------


def ticks_to_ohlcv(
    ticks: pl.DataFrame,
    time_column: str = "ts",
    price_column: str = "price",
    volume_column: str = "volume",
    every: str = "1m",
) -> pl.DataFrame:
    """
    Вход: таблица сделок/тиков (время, цена, объём в лотах или акциях).

    every: шаг группировки Polars (1m, 5m, 1h, 1d …).

    Выход: колонка begin (левый край интервала) + OHLCV.
    """
    if ticks.is_empty():
        return pl.DataFrame(
            schema={
                "begin": pl.Datetime("ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    df = ticks.sort(time_column)
    out = (
        df.group_by_dynamic(
            time_column,
            every=every,
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.first(price_column).alias("open"),
                pl.max(price_column).alias("high"),
                pl.min(price_column).alias("low"),
                pl.last(price_column).alias("close"),
                pl.sum(volume_column).alias("volume"),
            ]
        )
        .rename({time_column: "begin"})
    )
    return out


def example_ticks_frame(n: int = 500) -> pl.DataFrame:
    """Синтетические тики для локальной проверки ingestion (без API)."""
    import numpy as np
    from datetime import datetime, timedelta

    rng = np.random.default_rng(0)
    t0 = datetime(2025, 1, 2, 10, 0, 0)
    ts = [t0 + timedelta(seconds=int(i)) for i in range(n)]
    price = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    vol = rng.integers(1, 100, size=n).astype(float)
    return pl.DataFrame({"ts": ts, "price": price, "volume": vol})


# -----------------------------------------------------------------------------
# Orderlog: нормализация колонок (заготовка под агрессора и сторону)
# -----------------------------------------------------------------------------

ORDERLOG_COLUMNS_DOC = """
Ожидаемые имена после нормализации:
  ts — время (Datetime), price — цена, volume — объём,
  aggressor — +1 покупатель-агрессор, -1 продавец (опционально).
"""


def normalize_orderlog(df: pl.DataFrame) -> pl.DataFrame:
    """Алиасы колонок orderlog → ts, price, volume; опционально aggressor из buy."""
    colmap: dict[str, str] = {}
    lower = {c.lower(): c for c in df.columns}
    for want in ("ts", "time", "tradetime", "moment"):
        if want in lower:
            colmap[lower[want]] = "ts"
            break
    for want in ("price", "tradeprice", "p"):
        if want in lower:
            colmap[lower[want]] = "price"
            break
    for want in ("volume", "qty", "quantity", "v"):
        if want in lower:
            colmap[lower[want]] = "volume"
            break
    out = df.rename(colmap) if colmap else df.clone()
    if not any(c.lower() == "aggressor" for c in out.columns) and "buy" in lower:
        bcol = lower["buy"]
        out = out.with_columns(
            pl.when(pl.col(bcol)).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("aggressor")
        )
    return out
