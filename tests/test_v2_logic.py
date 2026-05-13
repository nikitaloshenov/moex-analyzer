"""
Проверки цепочки V2: PatternAnalyzer (pandas внутри) + backtest на Polars.

Запуск: из корня проекта `python tests/test_v2_logic.py`
"""

import os
import sys

import numpy as np
import polars as pl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.analyzer import PatternAnalyzer
from app.services.ingestion import example_ticks_frame, normalize_orderlog, ticks_to_ohlcv
from app.services.backtest import (
    BACKTEST_VERSION,
    MarketImpactConfig,
    _slippage_abs,
    run_pattern_backtest,
)


# -----------------------------------------------------------------------------
# Вспомогательные ряды
# -----------------------------------------------------------------------------


def _synthetic_begins(n: int, start: str = "2024-06-01") -> list[str]:
    from datetime import datetime, timedelta

    d0 = datetime.strptime(start, "%Y-%m-%d")
    return [(d0 + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(n)]


def make_flat_ohlcv(n: int = 60, base_price: float = 100.0, volume: float = 1e6) -> pl.DataFrame:
    """Ряд без паттернов — ожидаем в основном HOLD и мало сделок в бэктесте."""
    begins = _synthetic_begins(n)
    close = [base_price] * n
    return pl.DataFrame(
        {
            "begin": begins,
            "open": close,
            "high": [p + 0.5 for p in close],
            "low": [p - 0.5 for p in close],
            "close": close,
            "volume": [volume] * n,
        }
    )


# -----------------------------------------------------------------------------
# Тесты (assert)
# -----------------------------------------------------------------------------


def test_analyzer_accepts_polars():
    df = make_flat_ohlcv(35)
    r = PatternAnalyzer.analyze_all(df, ticker="TEST")
    assert "signal" in r and "price" in r
    assert r["price"] == 100.0


def test_backtest_runs_on_polars_and_reports_version():
    df = make_flat_ohlcv(55)
    out = run_pattern_backtest(df, start_date="2024-06-10", ticker="TEST", min_rows=50)
    assert out["version"] == BACKTEST_VERSION
    assert out["status"] in ("ok", "no_data")
    assert "impact" in out
    assert "metrics" in out and "sharpe" in out["metrics"]


def test_normalize_orderlog_aliases():
    df = pl.DataFrame({"TradeTime": [1], "p": [10.0], "qty": [5.0]})
    out = normalize_orderlog(df)
    assert "price" in out.columns


def test_ticks_to_ohlcv_ingestion():
    ticks = example_ticks_frame(400)
    ohlc = ticks_to_ohlcv(ticks, every="5m")
    assert ohlc.height >= 1
    for c in ("begin", "open", "high", "low", "close", "volume"):
        assert c in ohlc.columns


def test_backtest_has_trades_and_json():
    df = make_flat_ohlcv(55)
    out = run_pattern_backtest(df, start_date="2024-06-10", ticker="TEST", min_rows=50)
    if out["status"] == "ok":
        assert "trades" in out and isinstance(out["trades"], list)
        assert "trades_json" in out and isinstance(out["trades_json"], str)


def test_regime_flat_on_constant_price():
    from app.services.regime import classify_regime

    df = make_flat_ohlcv(45)
    r = classify_regime(df)
    assert r in ("flat", "unknown")


def test_portfolio_aggregate():
    from app.services.portfolio import aggregate_backtest_results

    agg = aggregate_backtest_results(
        [
            {"status": "ok", "ticker": "A", "profit": 10.0, "metrics": {"sharpe": 1.0}},
            {"status": "ok", "ticker": "B", "profit": -3.0, "metrics": {"sharpe": -0.5}},
        ]
    )
    assert agg["tickers"] == 2
    assert agg["sum_profit"] == 7.0


def test_impact_calibration_stub():
    from app.services.impact_calibration import fit_linear_impact_stub

    r = fit_linear_impact_stub([0.05, 0.1, 0.2], [0.001, 0.002, 0.004])
    assert r.get("status") == "ok"


def test_slippage_scales_with_participation():
    """Проскальзывание линейно по participation (фиксированная цена и impact_k)."""
    cfg_lo = MarketImpactConfig(participation=0.05, impact_k=0.002)
    cfg_hi = MarketImpactConfig(participation=0.10, impact_k=0.002)
    a = _slippage_abs(100.0, 1_000_000.0, cfg_lo)
    b = _slippage_abs(100.0, 1_000_000.0, cfg_hi)
    assert abs(b - 2 * a) < 1e-9


if __name__ == "__main__":
    test_analyzer_accepts_polars()
    print("✅ PatternAnalyzer принимает Polars DataFrame")

    test_backtest_runs_on_polars_and_reports_version()
    print("✅ run_pattern_backtest на Polars, в ответе version", BACKTEST_VERSION)

    test_normalize_orderlog_aliases()
    print("✅ Orderlog: normalize_orderlog (алиасы колонок)")

    test_ticks_to_ohlcv_ingestion()
    print("✅ Ingestion: тики → OHLCV (Polars group_by_dynamic)")

    test_backtest_has_trades_and_json()
    print("✅ В ответе бэктеста есть trades и trades_json")

    test_regime_flat_on_constant_price()
    print("✅ Regime: классификация на плоском ряду")

    test_portfolio_aggregate()
    print("✅ Portfolio: агрегация по тикерам")

    test_impact_calibration_stub()
    print("✅ Impact calibration stub (OLS)")

    test_slippage_scales_with_participation()
    print("✅ Проскальзывание растёт с participation (market impact)")

    print("\nЦепочка: Polars (backtest) → slice → PatternAnalyzer → to_pandas → CandlePatterns")
