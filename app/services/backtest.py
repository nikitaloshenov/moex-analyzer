from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import polars as pl
from app.core.config import settings
from app.services.analyzer import PatternAnalyzer
from app.services.regime import classify_regime, regime_allows_signal

BACKTEST_VERSION = "3.0_Sirius_Edition"

# -----------------------------------------------------------------------------
# Симуляция исполнения TWAP с Квадратичным Импактом (Модули M4 + M5)
# -----------------------------------------------------------------------------

def simulate_twap_execution(
    df_1m_slice: pl.DataFrame,
    is_buy: bool,
    order_size_q: float,
    impact_a: float = 0.03
) -> tuple[float, float]:
    """
    Модуль M4 & M5.
    Разбивает ордер order_size_q на 30 минутных слайсов.
    Вычисляет среднюю цену исполнения с учетом квадратичного импакта: 
    IS = a * x^2 * P * q
    
    Возвращает: (average_execution_price, total_implementation_shortfall)
    """
    if df_1m_slice.is_empty():
        return 0.0, 0.0

    # Количество доступных минутных свечей внутри 30-минутки (обычно 30)
    k = df_1m_slice.height
    q_slice = order_size_q / k  # Размер одного слайса (q)

    # Вытаскиваем массивы цен и объемов рынка
    prices = df_1m_slice["close"].to_numpy()
    market_volumes = df_1m_slice["volume"].to_numpy().astype(np.float64)

    total_is = 0.0
    execution_prices = []

    for price, market_vol in zip(prices, market_volumes):
        if market_vol <= 0:
            market_vol = 1.0  # Защита от деления на ноль

        # 1. Participation Rate (Доля нашего участия в этой минуте)
        x = q_slice / market_vol

        # 2. Формула квадратичного штрафа со слайдов: IS = a * x^2 * P * q
        is_penalty = impact_a * (x ** 2) * price * q_slice
        total_is += is_penalty

        # 3. Ухудшаем цену исполнения на величину импакта для этой минуты
        # Если покупаем — цена для нас растет, если продаем — падает
        if is_buy:
            exec_price = price + (impact_a * (x ** 2) * price)
        else:
            exec_price = price - (impact_a * (x ** 2) * price)
            
        execution_prices.append(exec_price)

    avg_exec_price = float(np.mean(execution_prices))
    return avg_exec_price, total_is


# -----------------------------------------------------------------------------
# Метрики (Сохранены для совместимости с UI)
# -----------------------------------------------------------------------------

def _max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0: return 0.0
    peak = np.maximum.accumulate(equity)
    base = np.where(np.abs(peak) < 1e-12, 1.0, peak)
    dd = (equity - peak) / base * 100.0
    return float(dd.min())

def _sharpe_sortino(pnls: list[float], entries: list[float]) -> tuple[float, float]:
    bm = settings.backtest_metrics
    rf_p = bm.risk_free_rate_annual / float(bm.periods_per_year)
    rets = np.array([p / abs(e) for p, e in zip(pnls, entries) if abs(e) > 1e-12], dtype=np.float64)
    if rets.size < 2: return 0.0, 0.0
    xs = rets - rf_p
    std = float(rets.std(ddof=1))
    if std < 1e-12: return 0.0, 0.0
    ann = np.sqrt(float(bm.periods_per_year))
    sharpe = float(xs.mean() / std * ann)
    downside = rets[rets < 0]
    return sharpe, (float(xs.mean() / float(downside.std(ddof=1)) * ann) if downside.size > 1 and float(downside.std(ddof=1)) > 1e-12 else sharpe)

# -----------------------------------------------------------------------------
# Главный бэктест-движок
# -----------------------------------------------------------------------------

def run_pattern_backtest(
    candles_30m: pl.DataFrame,
    candles_1m: pl.DataFrame,
    start_date: str,
    ticker: str = "DEFAULT",
) -> dict[str, Any]:
    """
    Execution-Aware Walk-Forward бэктест, адаптированный под критерии Сириуса.
    Прогоняет сигналы по 30-минуткам, симулирует исполнение TWAP внутри каждой сессии по 1-минуткам.
    """
    # Параметры ордеров из конфигов
    order_size_q = getattr(settings.execution, "order_size_lots", 10000.0)
    impact_a = getattr(settings.market_impact, "coefficient_a", 0.03)

    empty = {"ticker": ticker, "status": "no_data", "version": BACKTEST_VERSION, "trades": []}

    if candles_30m is None or candles_1m is None or candles_30m.is_empty():
        return empty

    n = candles_30m.height
    trade_pnls: list[float] = []
    trade_entries: list[float] = []
    total_shortfalls: list[float] = []
    trades: list[dict[str, Any]] = []

    # Идем по 30-минутным барам
    for i in range(20, n):  # Пропускаем первые 20 баров для прогрева индикаторов режима
        sub_pl = candles_30m.slice(0, i + 1)
        current_bar_time = candles_30m["begin"][i]

        # Проверка режима рынка
        regime = classify_regime(sub_pl)
        res = PatternAnalyzer.analyze_all(sub_pl, ticker=ticker)
        sig = res.get("signal", "HOLD")

        if settings.regime.enabled and not regime_allows_signal(sig, regime):
            continue

        if sig not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
            continue

        is_buy = "BUY" in sig

        # Вырезаем из 1-минутного датафрейма кусок, который принадлежит текущей 30-минутке
        # Для этого мы использовали поле signal_30m_bin на этапе Ingestion
        df_1m_slice = candles_1m.filter(pl.col("signal_30m_bin") == current_bar_time)
        if df_1m_slice.is_empty():
            continue

        # Симулируем TWAP вход (Получаем реальную цену исполнения и штраф)
        entry_price_net, shortfall = simulate_twap_execution(
            df_1m_slice, is_buy=is_buy, order_size_q=order_size_q, impact_a=impact_a
        )

        # Бумажная цена (середина рынка без импакта) для расчета идеального PnL
        mid_price = float(candles_30m["close"][i])

        # Симулируем выход по цене следующего 30-минутного бара (упрощенный контракт хакатона)
        if i + 1 >= n:
            break
        exit_price_mid = float(candles_30m["close"][i + 1])

        # Расчет Net PnL (Разница цен с учетом знака позиции * объем - штраф)
        direction = 1 if is_buy else -1
        pnl_points = (exit_price_mid - entry_price_net) * direction
        pnl_rub = pnl_points * order_size_q

        trade_pnls.append(pnl_rub)
        trade_entries.append(entry_price_net * order_size_q)
        total_shortfalls.append(shortfall)

        trades.append({
            "ticker": ticker,
            "entry_bar": i,
            "entry_date": str(current_bar_time),
            "side": "long" if is_buy else "short",
            "signal": sig,
            "regime": regime,
            "mid_price": round(mid_price, 4),
            "entry_price_net": round(entry_price_net, 4),
            "pnl_rub": round(pnl_rub, 2),
            "shortfall_rub": round(shortfall, 2),
        })

    # Расчет финальных агрегированных метрик для фронтенда
    total_trades = len(trade_pnls)
    if total_trades == 0: return empty

    wins = [t for t in trade_pnls if t > 0]
    winrate = (len(wins) / total_trades * 100)
    total_profit = float(sum(trade_pnls))
    
    eq = np.cumsum(np.array([0.0] + trade_pnls))
    max_dd_pct = _max_drawdown_pct(eq)
    sharpe, sortino = _sharpe_sortino(trade_pnls, trade_entries)

    return {
        "ticker": ticker,
        "winrate": winrate,
        "total": total_trades,
        "profit": total_profit,
        "status": "ok",
        "version": BACKTEST_VERSION,
        "metrics": {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd_pct,
            "total_shortfall_rub": float(sum(total_shortfalls)),
        },
        "trades": trades,
        "trades_json": json.dumps(trades, default=str),
    }