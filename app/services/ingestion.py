from __future__ import annotations
import polars as pl
from pathlib import Path

class DataIngestionService:
    def __init__(self, data_dir: str = "data"):
        """
        data_dir: путь к папке, где лежат CSV от организаторов.
        """
        self.data_dir = Path(data_dir)

    def load_clean_bars(self, file_30m: str, file_1m: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Загружает 30-минутные и 1-минутные бары, фильтрует торговую сессию MOEX
        и размечает временные интервалы для TWAP исполнения.
        
        Выход: (df_30m, df_1m)
        """
        path_30m = self.data_dir / file_30m
        path_1m = self.data_dir / file_1m

        if not path_30m.exists() or not path_1m.exists():
            raise FileNotFoundError(
                f"Критическая ошибка хакатона: файлы не найдены в {self.data_dir}.\n"
                f"Проверь наличие {file_30m} и {file_1m}"
            )

        # 1. Загрузка данных (Polars сам распарсит даты, если они в ISO-формате)
        df_30m = pl.read_csv(path_30m, try_parse_dates=True)
        df_1m = pl.read_csv(path_1m, try_parse_dates=True)

        # Унифицируем названия временной колонки к "begin" (как у тебя в тиках)
        for df in (df_30m, df_1m):
            for old_col in ["ts", "timestamp", "time", "datetime"]:
                if old_col in df.columns:
                    df.rename({old_col: "begin"})

        # Сортируем по времени (базовое требование для бэктестов)
        df_30m = df_30m.sort("begin")
        df_1m = df_1m.sort("begin")

        # 2. Фильтрация основной сессии MOEX (10:00 - 18:30) строго по слайдам
        df_30m = df_30m.filter(
            (df_30m["begin"].dt.time() >= pl.time(10, 0)) & 
            (df_30m["begin"].dt.time() <= pl.time(18, 30))
        )
        df_1m = df_1m.filter(
            (df_1m["begin"].dt.time() >= pl.time(10, 0)) & 
            (df_1m["begin"].dt.time() <= pl.time(18, 30))
        )

        # 3. Синхронизация таймфреймов через создание общего ключа группировки
        # Каждую минутную свечу мы "округляем" вниз до ближайших 30 минут.
        # Это свяжет 30 минутных свечей с одной сигнальной 30-минуткой.
        df_1m = df_1m.with_columns(
            pl.col("begin").dt.truncate("30m").alias("signal_30m_bin")
        )

        return df_30m, df_1m

# Сохраняем твою функцию на случай, если организаторы в финале дадут именно сырой orderlog
def ticks_to_ohlcv(
    ticks: pl.DataFrame,
    time_column: str = "ts",
    price_column: str = "price",
    volume_column: str = "volume",
    every: str = "1m",
) -> pl.DataFrame:
    if ticks.is_empty():
        return pl.DataFrame(schema={"begin": pl.Datetime("ms"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64})
    df = ticks.sort(time_column)
    return (
        df.group_by_dynamic(time_column, every=every, closed="left", label="left")
        .agg([
            pl.first(price_column).alias("open"),
            pl.max(price_column).alias("high"),
            pl.min(price_column).alias("low"),
            pl.last(price_column).alias("close"),
            pl.sum(volume_column).alias("volume"),
        ])
        .rename({time_column: "begin"})
    )