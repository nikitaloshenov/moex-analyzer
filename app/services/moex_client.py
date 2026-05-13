"""
Асинхронная загрузка свечей MOEX (moexalgo).

Роль: отдать pandas DataFrame истории; бэктест V2 сам переводит данные в Polars.
"""

import asyncio
import pandas as pd
from datetime import date, timedelta, datetime

from moexalgo import Ticker


# -----------------------------------------------------------------------------
# MoexService
# -----------------------------------------------------------------------------


class MoexService:
    @staticmethod
    async def get_historical_data(symbol: str, start_date: str, end_date: str):
        """Дневные свечи с запасом ~150 дней до start_date для прогрева индикаторов."""
        try:
            t = Ticker(symbol.upper())
            loop = asyncio.get_event_loop()

            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = (dt_start - timedelta(days=150)).strftime("%Y-%m-%d")

            candles = await loop.run_in_executor(
                None,
                lambda: t.candles(start=warmup_start, end=end_date, period="1D"),
            )

            if candles is not None and not candles.empty:
                return candles.sort_values("begin")
            return None
        except Exception as e:
            print(f"⚠️ Ошибка по тикеру {symbol}: {e}")
            return None

    @staticmethod
    async def get_data(symbol: str):
        """Онлайн: 10m свечи + дневки; ma200 по дневному close при длине ≥ 200."""
        try:
            t = Ticker(symbol.upper())
            today = date.today()
            loop = asyncio.get_event_loop()

            tasks = [
                loop.run_in_executor(
                    None,
                    lambda: t.candles(
                        start=str(today - timedelta(days=7)), end=str(today), period="10min"
                    ),
                ),
                loop.run_in_executor(
                    None,
                    lambda: t.candles(
                        start=str(today - timedelta(days=400)), end=str(today), period="1D"
                    ),
                ),
            ]

            candles_10m, candles_d1 = await asyncio.gather(*tasks)

            if candles_10m is None or candles_10m.empty:
                return None, None, None, None, None

            ma200_glob = None
            if candles_d1 is not None and len(candles_d1) >= 200:
                ma200_glob = candles_d1["close"].rolling(window=200).mean().iloc[-1]

            return candles_10m, candles_d1["low"].min(), candles_d1["high"].max(), ma200_glob, None
        except Exception:
            return None, None, None, None, None
