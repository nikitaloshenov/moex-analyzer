"""
Портфель: агрегация результатов по тикерам и заготовки под корреляцию / дневной стоп.

Полноценный общий equity по дням — следующий шаг (нужны даты сделок по всем тикерам).
"""

from __future__ import annotations

from typing import Any


# -----------------------------------------------------------------------------
# Агрегация
# -----------------------------------------------------------------------------


def aggregate_backtest_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Суммарный профит, средний Sharpe по тикерам со status=ok."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"status": "empty", "tickers": 0, "sum_profit": 0.0, "avg_sharpe": 0.0}
    sharps = [r.get("metrics", {}).get("sharpe", 0.0) or 0.0 for r in ok]
    return {
        "status": "ok",
        "tickers": len(ok),
        "sum_profit": float(sum(r.get("profit", 0.0) or 0.0 for r in ok)),
        "avg_sharpe": float(sum(sharps) / len(sharps)) if sharps else 0.0,
        "worst_ticker": min(ok, key=lambda x: x.get("profit", 0.0)).get("ticker"),
        "best_ticker": max(ok, key=lambda x: x.get("profit", 0.0)).get("ticker"),
    }


def correlation_guard_stub(
    _returns_by_ticker: dict[str, list[float]],
    max_abs: float | None = None,
) -> dict[str, Any]:
    """
    Заготовка: проверка max |корреляции| между рядами доходностей.

    Сейчас не считает корреляцию — вернёт ok=True (подключите numpy после сбора рядов).
    """
    lim = max_abs if max_abs is not None else 0.85
    return {
        "ok": True,
        "max_abs_corr": None,
        "limit": lim,
        "note": "stub: передайте матрицу дневных доходностей и вызовите np.corrcoef",
    }
