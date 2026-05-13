"""
Свечные паттерны и проверка объёма (pandas).

Роль: булевы признаки по последней (или заданной) свече DataFrame.
Используется только из PatternAnalyzer (вход — pandas после конвертации из Polars).
"""

import pandas as pd


# -----------------------------------------------------------------------------
# Класс паттернов
# -----------------------------------------------------------------------------


class CandlePatterns:
    """Математические правила pin / engulfing / railroad / inside / volume spike."""

    @staticmethod
    def is_pin_bar(df: pd.DataFrame, idx: int = -1) -> bool:
        """Пин-бар: маленькое тело, длинная нижняя или верхняя тень."""
        candle = df.iloc[idx]
        body = abs(candle["close"] - candle["open"])
        full_range = candle["high"] - candle["low"]

        if full_range == 0:
            return False

        lower_shadow = min(candle["open"], candle["close"]) - candle["low"]
        upper_shadow = candle["high"] - max(candle["open"], candle["close"])

        is_body_small = body < (full_range * 0.25)
        is_lower_long = lower_shadow > (full_range * 0.6)
        is_upper_long = upper_shadow > (full_range * 0.6)

        return is_body_small and (is_lower_long or is_upper_long)

    @staticmethod
    def is_engulfing(df: pd.DataFrame, idx: int = -1) -> bool:
        """Поглощение: тело текущей свечи шире тела предыдущей."""
        if len(df) < 2:
            return False

        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        curr_body_top = max(curr["open"], curr["close"])
        curr_body_bottom = min(curr["open"], curr["close"])
        prev_body_top = max(prev["open"], prev["close"])
        prev_body_bottom = min(prev["open"], prev["close"])

        return (curr_body_top > prev_body_top) and (curr_body_bottom < prev_body_bottom)

    @staticmethod
    def is_railroad_tracks(df: pd.DataFrame, idx: int = -1) -> bool:
        """Рельсы: два крупных тела разного направления, размеры близки."""
        if len(df) < 2:
            return False

        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        curr_body = abs(curr["close"] - curr["open"])
        prev_body = abs(prev["close"] - prev["open"])

        max_b = max(curr_body, prev_body)
        if max_b == 0:
            return False

        bodies_similar = abs(curr_body - prev_body) / max_b < 0.2
        different_dir = (curr["close"] > curr["open"] and prev["close"] < prev["open"]) or (
            curr["close"] < curr["open"] and prev["close"] > prev["open"]
        )

        return bodies_similar and different_dir

    @staticmethod
    def is_inside_bar(df: pd.DataFrame, idx: int = -1) -> bool:
        """Внутренний бар: диапазон текущей свечи внутри предыдущей."""
        if len(df) < 2:
            return False

        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        return (curr["high"] < prev["high"]) and (curr["low"] > prev["low"])

    @staticmethod
    def check_volume(df: pd.DataFrame, idx: int = -1, multiplier: float = 1.2) -> bool:
        """Текущий объём выше среднего за 20 баров с множителем."""
        if len(df) < 21:
            return False

        curr_vol = df.iloc[idx]["volume"]
        avg_vol = df["volume"].iloc[idx - 20 : idx].mean()

        return curr_vol > (avg_vol * multiplier)
