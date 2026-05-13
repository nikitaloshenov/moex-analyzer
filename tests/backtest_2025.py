"""
Скрипт массового бэктеста за 2025 год (V2: Polars + market impact внутри run_pattern_backtest).
"""

import asyncio
import sys
import os
from datetime import datetime

# --- Корень проекта в PYTHONPATH ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.moex_client import MoexService
from app.services.backtest import run_pattern_backtest

# -----------------------------------------------------------------------------
# Тикеры и запуск по одному году
# -----------------------------------------------------------------------------

TICKERS_40 = [
    "SBER", "GAZP", "LKOH", "NVTK", "GMKN", "YNDX", "ROSN", "MGNT", "TATN",
    "CHMF", "ALRS", "MTSS", "MAGN", "NLMK", "SNGS", "SNGSP", "VTBR", "TRNFP", "PHOR", "IRAO",
    "FLOT", "AFLT", "FEES", "CBOM", "MSNG", "AFKS", "MOEX", "HYDR", "PIKK", "MTLR",
    "UPRO", "RUAL", "LSRG", "BANEP", "SELG", "BSPB", "BELU", "POSI", "SVCB", "RNFT",
]


async def run_backtest(ticker):
    start_date = "2025-01-01"
    end_date = "2025-12-31"

    try:
        candles = await MoexService.get_historical_data(ticker, start_date, end_date)
    except Exception:
        return {"ticker": ticker, "status": "error"}

    return run_pattern_backtest(candles, start_date, ticker=ticker)


async def main():
    print("🚀 Бэктест V2 (Polars + ликвидность), 2025…")
    start_time = datetime.now()

    tasks = [run_backtest(t) for t in TICKERS_40]
    all_results = await asyncio.gather(*tasks)

    sorted_res = sorted(
        [r for r in all_results if r["status"] == "ok"],
        key=lambda x: x["profit"],
        reverse=True,
    )

    # --- Таблица результатов ---
    print("\n" + "=" * 80)
    print(f"{'Тикер':<8} | {'Winrate':<8} | {'Сделок':<7} | {'Profit Factor':<13} | {'Результат':<10}")
    print("-" * 80)

    for r in sorted_res:
        icon = "🟢" if r["profit"] > 0 else "🔴"
        print(
            f"{icon} {r['ticker']:<5} | {r['winrate']:>6.1f}% | {r['total']:>6} | "
            f"{r['pf']:>12.2f} | {r['profit']:>9.2f}"
        )

    print("=" * 80)
    if sorted_res:
        avg_winrate = sum(r["winrate"] for r in sorted_res) / len(sorted_res)
        total_market_profit = sum(r["profit"] for r in sorted_res)
        print(f"📈 Средний Winrate: {avg_winrate:.1f}%")
        print(f"💰 Суммарный результат (пункты): {total_market_profit:.2f}")
    print(f"⏱ Время: {(datetime.now() - start_time).total_seconds():.1f} сек")


if __name__ == "__main__":
    asyncio.run(main())
