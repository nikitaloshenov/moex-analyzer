from __future__ import annotations

from pathlib import Path
from datetime import time

import polars as pl


SESSION_START = time(10, 0)
SESSION_END = time(18, 30)


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=True)


def _feature_path(feature_dir: Path, feature: str) -> Path:
    return feature_dir / f"{feature}_.parquet"


def _normalize_time_column(df: pl.DataFrame) -> pl.DataFrame:
    for old in ("begin", "timestamp", "ts", "time", "datetime"):
        if old in df.columns:
            if old != "begin":
                df = df.rename({old: "begin"})
            return df
    raise ValueError("No timestamp column found. Expected begin/timestamp/ts/time/datetime.")


def _filter_session(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "begin" not in df.columns:
        return df
    return df.filter(
        (pl.col("begin").dt.time() >= SESSION_START)
        & (pl.col("begin").dt.time() <= SESSION_END)
    )


def _signal_end_bin(column: str = "begin", interval: str = "5m") -> pl.Expr:
    # end-bar logic: 10:01..10:05 -> 10:05, 10:06..10:10 -> 10:10
    minutes = int(interval.replace("m", ""))
    return (
        (pl.col(column) - pl.duration(microseconds=1)).dt.truncate(interval)
        + pl.duration(minutes=minutes)
    )


def load_feature_bars(
    feature_dir: str | Path,
    ticker: str,
    *,
    add_signal_bin: bool = False,
    signal_interval: str = "5m",
) -> pl.DataFrame:
    """
    Load hackathon feature-wise parquet data.

    Expected layout:
        open_.parquet
        high_.parquet
        low_.parquet
        close_.parquet
        buy_size_.parquet
        sell_size_.parquet
        optional best_bid_.parquet / best_ask_.parquet

    Output:
        begin, open, high, low, close, buy_size, sell_size, volume, best_bid, best_ask
    """
    feature_dir = Path(feature_dir)

    frames: list[pl.DataFrame] = []
    for feature in ("open", "high", "low", "close"):
        path = _feature_path(feature_dir, feature)
        if not path.exists():
            raise FileNotFoundError(f"Required feature file not found: {path}")
        frames.append(
            pl.read_parquet(path, columns=["timestamp", ticker])
            .rename({"timestamp": "begin", ticker: feature})
        )

    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="begin", how="inner")

    for feature in ("buy_size", "sell_size"):
        path = _feature_path(feature_dir, feature)
        if path.exists():
            extra = (
                pl.read_parquet(path, columns=["timestamp", ticker])
                .rename({"timestamp": "begin", ticker: feature})
            )
            out = out.join(extra, on="begin", how="left")
        else:
            out = out.with_columns(pl.lit(0.0).alias(feature))

    for feature in ("best_bid", "best_ask"):
        path = _feature_path(feature_dir, feature)
        if path.exists():
            quote = (
                pl.read_parquet(path, columns=["timestamp", ticker])
                .rename({"timestamp": "begin", ticker: feature})
            )
            out = out.join(quote, on="begin", how="left")

    out = (
        out.with_columns(
            [
                pl.col("buy_size").fill_null(0.0),
                pl.col("sell_size").fill_null(0.0),
                (pl.col("buy_size").fill_null(0.0) + pl.col("sell_size").fill_null(0.0)).alias("volume"),
                pl.lit(ticker).alias("seccode"),
            ]
        )
        .drop_nulls(["begin", "open", "high", "low", "close"])
        .sort("begin")
    )

    out = _filter_session(out)

    if add_signal_bin:
        out = out.with_columns(
            [
                _signal_end_bin("begin", signal_interval).alias("signal_5m_bin"),
                _signal_end_bin("begin", signal_interval).alias("signal_30m_bin"),  # legacy compatibility
            ]
        )

    return out


def load_feature_pair(
    data_root: str | Path,
    ticker: str,
    *,
    dir_1m: str = "is_features_1_min_hackaton",
    dir_5m: str = "is_features_5_min_hackaton",
    signal_interval: str = "5m",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    data_root = Path(data_root)
    bars_5m = load_feature_bars(
        data_root / dir_5m,
        ticker,
        add_signal_bin=False,
        signal_interval=signal_interval,
    )
    bars_1m = load_feature_bars(
        data_root / dir_1m,
        ticker,
        add_signal_bin=True,
        signal_interval=signal_interval,
    )
    return bars_5m, bars_1m


def discover_feature_tickers(
    data_root: str | Path,
    *,
    dir_5m: str = "is_features_5_min_hackaton",
    feature: str = "close",
) -> list[str]:
    path = Path(data_root) / dir_5m / f"{feature}_.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Ticker discovery file not found: {path}")

    timestamp_columns = {"timestamp", "begin", "ts", "time", "datetime"}
    schema = pl.scan_parquet(path).collect_schema()
    return [
        column
        for column in schema.names()
        if column.lower() not in timestamp_columns
    ]


class DataIngestionService:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_clean_bars(
        self,
        file_5m: str,
        file_1m: str,
        *,
        signal_interval: str = "5m",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        path_5m = self.data_dir / file_5m
        path_1m = self.data_dir / file_1m

        if not path_5m.exists() or not path_1m.exists():
            raise FileNotFoundError(f"Bars not found: {path_5m}, {path_1m}")

        bars_5m = _normalize_time_column(_read_table(path_5m)).sort("begin")
        bars_1m = _normalize_time_column(_read_table(path_1m)).sort("begin")

        bars_5m = _filter_session(bars_5m)
        bars_1m = _filter_session(bars_1m).with_columns(
            [
                _signal_end_bin("begin", signal_interval).alias("signal_5m_bin"),
                _signal_end_bin("begin", signal_interval).alias("signal_30m_bin"),
            ]
        )

        return bars_5m, bars_1m


def ticks_to_ohlcv(
    ticks: pl.DataFrame,
    *,
    time_column: str = "ts",
    price_column: str = "price",
    volume_column: str = "volume",
    every: str = "1m",
) -> pl.DataFrame:
    if ticks.is_empty():
        return pl.DataFrame(
            schema={
                "begin": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    return (
        ticks.sort(time_column)
        .group_by_dynamic(time_column, every=every, closed="left", label="right")
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
        .drop_nulls(["open", "high", "low", "close"])
    )
