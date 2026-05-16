from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import polars as pl

from app.core.config import settings
from app.services.analyzer import PatternAnalyzer


BACKTEST_VERSION = "formal_m1_m5_clean"
ExecutionMode = Literal["twap", "optimal"]

FORMAL_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "signal_5min": ("bar_end_ts", "seccode", "value"),
    "backtest_baseline": ("bar_end_ts", "seccode", "pos", "pnl_mid", "cum_pnl_mid"),
    "impact_model": ("bar_end_ts", "seccode", "volume_mkt", "a"),
    "execution_schedule": (
        "bar_end_ts_5min",
        "bar_end_ts_1min",
        "seccode",
        "q_slice",
        "participation_rate",
        "impact_cost_rel",
    ),
    "execution_summary": (
        "bar_end_ts_5min",
        "seccode",
        "Q_executed",
        "vwap_fill",
        "twap_bench",
        "implementation_shortfall",
        "pnl_mid",
        "pnl_net",
    ),
}


def _ts_int(value: Any) -> int:
    if hasattr(value, "timestamp"):
        return int(value.timestamp() * 1_000_000_000)
    try:
        return int(value)
    except Exception:
        return 0


def _empty_result(ticker: str, status: str = "no_data") -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": status,
        "version": BACKTEST_VERSION,
        "profit": 0.0,
        "total": 0,
        "winrate": 0.0,
        "pnl_mid": 0.0,
        "pnl_net": 0.0,
        "cum_pnl_mid": 0.0,
        "cum_pnl_net": 0.0,
        "implementation_shortfall": 0.0,
        "signal_5min": [],
        "backtest_baseline": [],
        "impact_model": [],
        "execution_schedule": [],
        "execution_summary": [],
        "trades": [],
        "trades_json": "[]",
        "metrics": {
            "total_shortfall_rub": 0.0,
            "shortfall_ratio": 0.0,
            "hit_rate_mid": 0.0,
            "sharpe_pnl_net": 0.0,
            "is_reduction_pct": 0.0,
            "avg_participation_rate": 0.0,
            "max_participation_rate": 0.0,
            "avg_Q_executed": 0.0,
            "max_Q_executed": 0.0,
            "approx_turnover_rub": 0.0,
            "pnl_net_pct_of_turnover": 0.0,
            "implementation_shortfall_pct_of_turnover": 0.0,
        },
    }


def validate_formal_output(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}

    for table, required_columns in FORMAL_TABLE_COLUMNS.items():
        data = result.get(table)
        if isinstance(data, pl.DataFrame):
            columns = set(data.columns)
            row_count = data.height
        elif isinstance(data, list):
            columns = set().union(*(row.keys() for row in data if isinstance(row, dict))) if data else set()
            row_count = len(data)
        else:
            columns = set()
            row_count = 0

        missing = [column for column in required_columns if column not in columns]
        report[table] = {
            "ok": not missing,
            "rows": row_count,
            "missing": missing,
        }

    return report


def _safe_price(df: pl.DataFrame, column: str, idx: int) -> float:
    try:
        value = float(df[column][idx])
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _sharpe_from_pnl(pnls: list[float]) -> float:
    arr = np.array([p for p in pnls if np.isfinite(p)], dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * np.sqrt(arr.size))


def _participation_metrics(schedule: list[dict[str, Any]]) -> dict[str, float]:
    values = [
        float(row.get("participation_rate", 0.0))
        for row in schedule
        if np.isfinite(float(row.get("participation_rate", 0.0)))
    ]
    if not values:
        return {"avg_participation_rate": 0.0, "max_participation_rate": 0.0}
    return {
        "avg_participation_rate": float(np.mean(values)),
        "max_participation_rate": float(np.max(values)),
    }


def _estimate_impact_a_from_past(df_1m_past: pl.DataFrame, fallback_a: float) -> float:
    mi = settings.market_impact

    if not getattr(mi, "estimate_from_spread", False):
        return float(fallback_a)

    if df_1m_past.is_empty() or "best_bid" not in df_1m_past.columns or "best_ask" not in df_1m_past.columns:
        return float(fallback_a)

    row = df_1m_past.select(
        ((pl.col("best_ask") - pl.col("best_bid")) / pl.col("close"))
        .filter((pl.col("best_ask") > 0) & (pl.col("best_bid") > 0) & (pl.col("close") > 0))
        .median()
        .alias("spread_rel")
    )

    if row.is_empty() or row["spread_rel"][0] is None:
        return float(fallback_a)

    spread_rel = float(row["spread_rel"][0])
    if not np.isfinite(spread_rel) or spread_rel <= 0:
        return float(fallback_a)

    return float(np.clip(max(fallback_a, spread_rel * 10.0), 0.01, 0.05))


def build_impact_model(
    bars_1m: pl.DataFrame,
    *,
    seccode: str,
    fallback_a: float | None = None,
) -> pl.DataFrame:
    if bars_1m is None or bars_1m.is_empty():
        return pl.DataFrame(
            schema={
                "bar_end_ts": pl.Int64,
                "seccode": pl.Utf8,
                "volume_mkt": pl.Float64,
                "a": pl.Float64,
            }
        )

    a = float(fallback_a if fallback_a is not None else getattr(settings.market_impact, "coefficient_a", 0.03))

    return bars_1m.select(
        [
            pl.col("begin").map_elements(_ts_int, return_dtype=pl.Int64).alias("bar_end_ts"),
            pl.lit(seccode).alias("seccode"),
            pl.col("volume").fill_null(0.0).cast(pl.Float64).alias("volume_mkt"),
            pl.lit(a).cast(pl.Float64).alias("a"),
        ]
    )


def build_backtest_baseline(
    bars_5m: pl.DataFrame,
    signal_5min: pl.DataFrame,
    *,
    seccode: str,
    threshold: float = 0.0,
    order_size_q: float = 1.0,
) -> pl.DataFrame:
    if bars_5m is None or bars_5m.is_empty() or signal_5min.is_empty():
        return pl.DataFrame(
            schema={
                "bar_end_ts": pl.Int64,
                "seccode": pl.Utf8,
                "pos": pl.Float64,
                "pnl_mid": pl.Float64,
                "cum_pnl_mid": pl.Float64,
            }
        )

    values = signal_5min["value"].to_list()
    rows: list[dict[str, Any]] = []
    cum = 0.0

    for i in range(1, bars_5m.height):
        prev_signal = values[i - 1]

        if prev_signal is None or not np.isfinite(float(prev_signal)) or abs(float(prev_signal)) <= threshold:
            pos = 0.0
        else:
            pos = float(np.clip(prev_signal, -1.0, 1.0))

        entry = _safe_price(bars_5m, "open", i)
        exit_ = _safe_price(bars_5m, "close", i)

        pnl_mid = pos * (exit_ - entry) * float(order_size_q) if entry > 0 and exit_ > 0 else 0.0
        cum += pnl_mid

        rows.append(
            {
                "bar_end_ts": _ts_int(bars_5m["begin"][i]),
                "seccode": seccode,
                "pos": pos,
                "pnl_mid": pnl_mid,
                "cum_pnl_mid": cum,
            }
        )

    return pl.DataFrame(rows)


def _execution_slices(bars_1m: pl.DataFrame, bar_5m_time: Any) -> pl.DataFrame:
    if bars_1m.is_empty():
        return bars_1m

    if "signal_5m_bin" in bars_1m.columns:
        return bars_1m.filter(pl.col("signal_5m_bin") == bar_5m_time)

    if "signal_30m_bin" in bars_1m.columns:
        return bars_1m.filter(pl.col("signal_30m_bin") == bar_5m_time)

    return bars_1m.filter(pl.col("begin") == bar_5m_time)


def _target_order_size(price: float, target_notional_rub: float | None = None) -> float:
    q = float(getattr(settings.execution, "order_size_lots", 10000.0))

    if target_notional_rub is not None and price > 0:
        q = max(float(target_notional_rub) / price, 1.0)

    max_notional = getattr(settings.risk, "max_position_rub", None)
    if max_notional is not None and price > 0:
        q = min(q, float(max_notional) / price)

    return max(q, 1.0)


def _execution_order_size(
    *,
    fixed_q: float,
    entry_price: float,
    df_1m_slice: pl.DataFrame,
) -> dict[str, Any]:
    dynamic = bool(getattr(settings.execution, "dynamic_order_sizing", False))
    target_participation = float(getattr(settings.execution, "target_participation_rate", 0.01))
    max_notional = getattr(settings.execution, "max_order_notional_rub", None)
    min_order_size = float(getattr(settings.execution, "min_order_size_lots", 1.0))

    q_requested = float(fixed_q)
    q_liquidity_cap: float | None = None
    q_notional_cap: float | None = None
    q_executed = q_requested

    if dynamic:
        if "volume_mkt" in df_1m_slice.columns:
            volume_expr = pl.col("volume_mkt")
        else:
            volume_expr = pl.col("volume")

        row = df_1m_slice.select(
            volume_expr.fill_null(0.0).cast(pl.Float64).sum().alias("slice_volume_sum")
        )
        slice_volume_sum = float(row["slice_volume_sum"][0] or 0.0) if not row.is_empty() else 0.0
        q_liquidity_cap = max(target_participation * slice_volume_sum, 0.0)

        caps = [q_requested, q_liquidity_cap]
        if max_notional is not None and entry_price > 0:
            q_notional_cap = max(float(max_notional) / entry_price, 0.0)
            caps.append(q_notional_cap)

        q_executed = float(min(caps))

    return {
        "Q_requested": q_requested,
        "Q_executed": q_executed,
        "q_liquidity_cap": q_liquidity_cap,
        "q_notional_cap": q_notional_cap,
        "dynamic_order_sizing": dynamic,
        "target_participation_rate": target_participation,
        "min_order_size_lots": min_order_size,
    }


def _simulate_execution_schedule(
    df_1m_slice: pl.DataFrame,
    *,
    bar_end_ts_5min: int,
    seccode: str,
    is_buy: bool,
    order_size_q: float,
    impact_a: float,
    mode: ExecutionMode = "twap",
    max_participation: float = 0.30,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if df_1m_slice.is_empty() or order_size_q <= 0:
        return {
            "Q_executed": 0.0,
            "vwap_fill": 0.0,
            "twap_bench": 0.0,
            "implementation_shortfall": 0.0,
            "fill_price": 0.0,
        }, []

    rows = df_1m_slice.select(["begin", "close", "volume"]).drop_nulls(["begin", "close"])
    if rows.is_empty():
        return {
            "Q_executed": 0.0,
            "vwap_fill": 0.0,
            "twap_bench": 0.0,
            "implementation_shortfall": 0.0,
            "fill_price": 0.0,
        }, []

    times = rows["begin"].to_list()
    prices = rows["close"].to_numpy().astype(float)
    volumes_raw = rows["volume"].fill_null(0.0).to_numpy().astype(float)

    valid = np.isfinite(prices) & (prices > 0)
    times = [t for t, ok in zip(times, valid) if ok]
    prices = prices[valid]
    volumes_raw = volumes_raw[valid]

    if prices.size == 0:
        return {
            "Q_executed": 0.0,
            "vwap_fill": 0.0,
            "twap_bench": 0.0,
            "implementation_shortfall": 0.0,
            "fill_price": 0.0,
        }, []

    safe_volumes = np.where(np.isfinite(volumes_raw) & (volumes_raw > 0), volumes_raw, 0.0)
    volume_sum = float(safe_volumes.sum())

    if mode == "optimal":
        if volume_sum > 0:
            weights = safe_volumes / volume_sum
        else:
            weights = np.full(prices.size, 1.0 / prices.size)
    else:
        weights = np.full(prices.size, 1.0 / prices.size)

    planned_q_slices = float(order_size_q) * weights
    q_caps = float(max_participation) * safe_volumes
    q_slices = np.minimum(planned_q_slices, q_caps)
    actual_order_size_q = float(q_slices.sum())

    if actual_order_size_q <= 1e-12:
        return {
            "Q_executed": 0.0,
            "vwap_fill": 0.0,
            "twap_bench": float(np.mean(prices)),
            "implementation_shortfall": 0.0,
            "fill_price": 0.0,
        }, []

    participation = np.divide(
        q_slices,
        safe_volumes,
        out=np.zeros_like(q_slices, dtype=float),
        where=safe_volumes > 0,
    )

    impact_cost_rel = float(impact_a) * participation
    fill_prices = prices * (1.0 + impact_cost_rel if is_buy else 1.0 - impact_cost_rel)

    implementation_shortfall = float(np.sum(impact_cost_rel * prices * q_slices))
    fill_price = float(np.sum(fill_prices * q_slices) / actual_order_size_q)
    twap_bench = float(np.mean(prices))

    schedule: list[dict[str, Any]] = []
    for ts, price, volume, q_slice, part, impact_rel, fill in zip(
        times, prices, safe_volumes, q_slices, participation, impact_cost_rel, fill_prices
    ):
        schedule.append(
            {
                "bar_end_ts_5min": int(bar_end_ts_5min),
                "bar_end_ts_1min": _ts_int(ts),
                "seccode": seccode,
                "q_slice": float(q_slice),
                "volume_mkt": float(volume),
                "participation_rate": float(part),
                "impact_cost_rel": float(impact_rel),
                "mid_price": float(price),
                "fill_price": float(fill),
                "side": "buy" if is_buy else "sell",
                "mode": mode,
                "a": float(impact_a),
            }
        )

    return {
        "Q_executed": float(actual_order_size_q),
        "vwap_fill": float(fill_price),
        "twap_bench": float(twap_bench),
        "implementation_shortfall": float(implementation_shortfall),
        "fill_price": float(fill_price),
    }, schedule


def run_pattern_backtest(
    candles_30m: pl.DataFrame,
    candles_1m: pl.DataFrame,
    start_date: str = "",
    ticker: str = "DEFAULT",
    *,
    execution_mode: ExecutionMode = "twap",
    target_notional_rub: float | None = None,
    signal_threshold: float = 0.0,
) -> dict[str, Any]:
    bars_5m = candles_30m
    bars_1m = candles_1m

    if bars_5m is None or bars_1m is None or bars_5m.is_empty() or bars_1m.is_empty():
        return _empty_result(ticker)

    bars_5m = bars_5m.sort("begin")
    bars_1m = bars_1m.sort("begin")

    impact_a_default = float(getattr(settings.market_impact, "coefficient_a", 0.03))
    signal_5min_df = PatternAnalyzer.build_signal_5min(bars_5m, ticker=ticker)

    ref_price = float(bars_5m["close"].drop_nulls().median() or 1.0)
    base_q = _target_order_size(ref_price, target_notional_rub)

    baseline_df = build_backtest_baseline(
        bars_5m,
        signal_5min_df,
        seccode=ticker,
        threshold=signal_threshold,
        order_size_q=base_q,
    )
    impact_model_df = build_impact_model(bars_1m, seccode=ticker, fallback_a=impact_a_default)

    signal_values = signal_5min_df["value"].to_list()

    execution_schedule: list[dict[str, Any]] = []
    execution_summary: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    pnl_mid_total = 0.0
    pnl_net_total = 0.0
    is_total = 0.0
    turnover_total = 0.0
    pnl_mid_series: list[float] = []
    pnl_net_series: list[float] = []

    for i in range(1, bars_5m.height):
        prev_signal = signal_values[i - 1]

        if prev_signal is None or not np.isfinite(float(prev_signal)) or abs(float(prev_signal)) <= signal_threshold:
            continue

        pos = float(np.clip(prev_signal, -1.0, 1.0))
        is_buy = pos > 0

        bar_time = bars_5m["begin"][i]
        bar_end_ts_5min = _ts_int(bar_time)

        entry_open = _safe_price(bars_5m, "open", i)
        exit_close = _safe_price(bars_5m, "close", i)

        if entry_open <= 0 or exit_close <= 0:
            continue

        q_requested = _target_order_size(entry_open, target_notional_rub)
        slices = _execution_slices(bars_1m, bar_time)

        if slices.is_empty():
            continue

        sizing = _execution_order_size(
            fixed_q=q_requested,
            entry_price=entry_open,
            df_1m_slice=slices,
        )
        q = float(sizing["Q_executed"])

        if q < float(sizing["min_order_size_lots"]):
            continue

        past_1m = bars_1m.filter(pl.col("begin") < bar_time).tail(120)
        impact_a = _estimate_impact_a_from_past(past_1m, impact_a_default)

        twap_summary, twap_schedule_rows = _simulate_execution_schedule(
            slices,
            bar_end_ts_5min=bar_end_ts_5min,
            seccode=ticker,
            is_buy=is_buy,
            order_size_q=q,
            impact_a=impact_a,
            mode="twap",
            max_participation=0.30,
        )

        optimal_summary, optimal_schedule_rows = _simulate_execution_schedule(
            slices,
            bar_end_ts_5min=bar_end_ts_5min,
            seccode=ticker,
            is_buy=is_buy,
            order_size_q=q,
            impact_a=impact_a,
            mode="optimal",
            max_participation=0.30,
        )

        if execution_mode == "optimal":
            actual_summary = optimal_summary
            actual_schedule_rows = optimal_schedule_rows
        else:
            actual_summary = twap_summary
            actual_schedule_rows = twap_schedule_rows

        if not actual_schedule_rows:
            continue

        direction = 1.0 if is_buy else -1.0
        q_twap = float(twap_summary["Q_executed"])
        q_optimal = float(optimal_summary["Q_executed"])
        q_actual = float(actual_summary["Q_executed"])

        pnl_mid_twap = direction * (exit_close - entry_open) * q_twap
        pnl_mid_optimal = direction * (exit_close - entry_open) * q_optimal
        pnl_mid = pnl_mid_optimal if execution_mode == "optimal" else pnl_mid_twap

        twap_implementation_shortfall = float(twap_summary["implementation_shortfall"])
        optimal_implementation_shortfall = float(optimal_summary["implementation_shortfall"])
        implementation_shortfall = (
            optimal_implementation_shortfall
            if execution_mode == "optimal"
            else twap_implementation_shortfall
        )

        pnl_net_twap = pnl_mid_twap - twap_implementation_shortfall
        pnl_net_optimal = pnl_mid_optimal - optimal_implementation_shortfall
        pnl_net = pnl_net_optimal if execution_mode == "optimal" else pnl_net_twap

        pnl_mid_total += pnl_mid
        pnl_net_total += pnl_net
        is_total += implementation_shortfall
        turnover_total += abs(q_actual * entry_open)

        pnl_mid_series.append(float(pnl_mid))
        pnl_net_series.append(float(pnl_net))

        is_reduction_trade = (
            (twap_implementation_shortfall - optimal_implementation_shortfall)
            / twap_implementation_shortfall
            * 100.0
            if twap_implementation_shortfall > 1e-12
            else 0.0
        )

        # Formal field name: vwap_fill is the achieved volume-weighted fill price
        # of the selected schedule. It is not a VWAP execution mode.
        summary_row = {
            "bar_end_ts_5min": bar_end_ts_5min,
            "seccode": ticker,
            "Q_requested": float(sizing["Q_requested"]),
            "Q_executed": float(q_actual),
            "q_liquidity_cap": sizing["q_liquidity_cap"],
            "q_notional_cap": sizing["q_notional_cap"],
            "dynamic_order_sizing": bool(sizing["dynamic_order_sizing"]),
            "target_participation_rate": float(sizing["target_participation_rate"]),
            "execution_mode": execution_mode,
            "vwap_fill": float(actual_summary["vwap_fill"]),
            "twap_bench": float(actual_summary["twap_bench"]),
            "twap_implementation_shortfall": twap_implementation_shortfall,
            "optimal_implementation_shortfall": optimal_implementation_shortfall,
            "implementation_shortfall": implementation_shortfall,
            "is_reduction_pct": float(is_reduction_trade),
            "pnl_mid": float(pnl_mid),
            "pnl_net": float(pnl_net),
            "pnl_net_twap": float(pnl_net_twap),
            "pnl_net_optimal": float(pnl_net_optimal),
            "mode": execution_mode,
            "a": float(impact_a),
        }

        execution_summary.append(summary_row)
        execution_schedule.extend(actual_schedule_rows)

        trades.append(
            {
                "ticker": ticker,
                "entry_date": str(bar_time),
                "bar_end_ts": bar_end_ts_5min,
                "side": "long" if is_buy else "short",
                "signal_value": float(pos),
                "entry_mid": float(entry_open),
                "exit_mid": float(exit_close),
                "Q_requested": float(sizing["Q_requested"]),
                "Q_executed": float(q_actual),
                "q_liquidity_cap": sizing["q_liquidity_cap"],
                "q_notional_cap": sizing["q_notional_cap"],
                "dynamic_order_sizing": bool(sizing["dynamic_order_sizing"]),
                "pnl_mid": float(pnl_mid),
                "pnl_net": float(pnl_net),
                "pnl_net_twap": float(pnl_net_twap),
                "pnl_net_optimal": float(pnl_net_optimal),
                "twap_implementation_shortfall": float(twap_implementation_shortfall),
                "implementation_shortfall": float(implementation_shortfall),
                "optimal_implementation_shortfall": float(optimal_implementation_shortfall),
                "is_reduction_pct": float(is_reduction_trade),
                "shortfall_rub": float(implementation_shortfall),
                "execution_mode": execution_mode,
            }
        )

    total = len(trades)

    if total == 0:
        result = _empty_result(ticker, status="no_trades")
        result["signal_5min"] = signal_5min_df.to_dicts()
        result["backtest_baseline"] = baseline_df.to_dicts()
        result["impact_model"] = impact_model_df.to_dicts()
        return result

    wins = [t for t in trades if float(t["pnl_net"]) > 0]
    hit_mid = [t for t in trades if float(t["pnl_mid"]) > 0]

    shortfall_ratio = is_total / max(abs(pnl_mid_total), 1.0)

    participation_stats = _participation_metrics(execution_schedule)
    sharpe_pnl_net = _sharpe_from_pnl(pnl_net_series)

    twap_is_total = float(sum(float(row.get("twap_implementation_shortfall", 0.0)) for row in execution_summary))
    optimal_is_total = float(sum(float(row.get("optimal_implementation_shortfall", 0.0)) for row in execution_summary))
    q_executed_values = [
        float(row.get("Q_executed", 0.0))
        for row in execution_summary
        if np.isfinite(float(row.get("Q_executed", 0.0)))
    ]

    is_reduction_pct = (
        (twap_is_total - optimal_is_total) / twap_is_total * 100.0
        if twap_is_total > 1e-12
        else 0.0
    )
    pnl_net_pct_of_turnover = pnl_net_total / turnover_total * 100.0 if turnover_total > 1e-12 else 0.0
    is_pct_of_turnover = is_total / turnover_total * 100.0 if turnover_total > 1e-12 else 0.0

    return {
        "ticker": ticker,
        "status": "ok",
        "version": BACKTEST_VERSION,
        "profit": float(pnl_net_total),
        "total": total,
        "winrate": len(wins) / total * 100.0,
        "pnl_mid": float(pnl_mid_total),
        "pnl_net": float(pnl_net_total),
        "cum_pnl_mid": float(pnl_mid_total),
        "cum_pnl_net": float(pnl_net_total),
        "implementation_shortfall": float(is_total),
        "signal_5min": signal_5min_df.to_dicts(),
        "backtest_baseline": baseline_df.to_dicts(),
        "impact_model": impact_model_df.to_dicts(),
        "execution_schedule": execution_schedule,
        "execution_summary": execution_summary,
        "trades": trades,
        "trades_json": json.dumps(trades, default=str),
        "metrics": {
            "total_shortfall_rub": float(is_total),
            "shortfall_ratio": float(shortfall_ratio),
            "hit_rate_mid": len(hit_mid) / total * 100.0,
            "sharpe_pnl_net": float(sharpe_pnl_net),
            "is_reduction_pct": float(is_reduction_pct),
            "avg_participation_rate": float(participation_stats["avg_participation_rate"]),
            "max_participation_rate": float(participation_stats["max_participation_rate"]),
            "avg_Q_executed": float(np.mean(q_executed_values)) if q_executed_values else 0.0,
            "max_Q_executed": float(np.max(q_executed_values)) if q_executed_values else 0.0,
            "approx_turnover_rub": float(turnover_total),
            "pnl_net_pct_of_turnover": float(pnl_net_pct_of_turnover),
            "implementation_shortfall_pct_of_turnover": float(is_pct_of_turnover),
        },
    }
