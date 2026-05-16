from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl

from app.core.config import settings
from app.services.backtest import run_pattern_backtest, validate_formal_output
from app.services.ingestion import load_feature_pair


DEFAULT_Q_GRID = (100.0, 500.0, 1000.0, 5000.0, 10000.0, 25000.0)


def _shortfall_zone(shortfall_ratio: float) -> str:
    if shortfall_ratio < 0.5:
        return "SAFE"
    if shortfall_ratio < 1.0:
        return "ACCEPTABLE"
    if shortfall_ratio < 2.0:
        return "DANGEROUS"
    return "BROKEN"


def _records_frame(records: list[dict[str, Any]], columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, list(columns)] if not frame.empty else pd.DataFrame(columns=list(columns))


def _pl_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return frame.to_pandas() if frame is not None and not frame.is_empty() else pd.DataFrame(columns=frame.columns if frame is not None else [])


def _run_with_execution_settings(
    bars_5m: pl.DataFrame,
    bars_1m: pl.DataFrame,
    *,
    ticker: str,
    execution_mode: str,
    target_notional_rub: float | None,
    signal_threshold: float,
    order_size_lots: float | None,
    dynamic_order_sizing: bool | None,
    target_participation_rate: float | None,
    max_order_notional_rub: float | None,
    min_order_size_lots: float | None,
) -> dict[str, Any]:
    original_execution = {
        "order_size_lots": settings.execution.order_size_lots,
        "dynamic_order_sizing": settings.execution.dynamic_order_sizing,
        "target_participation_rate": settings.execution.target_participation_rate,
        "max_order_notional_rub": settings.execution.max_order_notional_rub,
        "min_order_size_lots": settings.execution.min_order_size_lots,
    }

    try:
        if order_size_lots is not None:
            settings.execution.order_size_lots = float(order_size_lots)
        if dynamic_order_sizing is not None:
            settings.execution.dynamic_order_sizing = bool(dynamic_order_sizing)
        if target_participation_rate is not None:
            settings.execution.target_participation_rate = float(target_participation_rate)
        if max_order_notional_rub is not None:
            settings.execution.max_order_notional_rub = float(max_order_notional_rub)
        if min_order_size_lots is not None:
            settings.execution.min_order_size_lots = float(min_order_size_lots)

        return run_pattern_backtest(
            bars_5m,
            bars_1m,
            ticker=ticker,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            target_notional_rub=target_notional_rub,
            signal_threshold=signal_threshold,
        )
    finally:
        for key, value in original_execution.items():
            setattr(settings.execution, key, value)


class Pipeline:
    @staticmethod
    def run(
        *,
        data_root: str | Path = "insample_data",
        ticker: str = "SBER",
        bars: int = 300,
        execution_mode: str = "optimal",
        order_size_lots: float | None = None,
        dynamic_order_sizing: bool | None = True,
        target_participation_rate: float | None = 0.01,
        target_notional_rub: float | None = None,
        max_order_notional_rub: float | None = None,
        min_order_size_lots: float | None = None,
        signal_threshold: float = 0.0,
        q_grid: Iterable[float] = DEFAULT_Q_GRID,
    ) -> dict[str, Any]:
        ticker = ticker.upper()
        bars_5m, bars_1m = load_feature_pair(data_root, ticker)

        if bars > 0:
            bars_5m = bars_5m.head(bars)
            if not bars_5m.is_empty():
                last_ts = bars_5m["begin"][-1]
                bars_1m = bars_1m.filter(pl.col("begin") <= last_ts)

        requested_q = float(order_size_lots if order_size_lots is not None else settings.execution.order_size_lots)
        dynamic = bool(dynamic_order_sizing if dynamic_order_sizing is not None else settings.execution.dynamic_order_sizing)
        target_participation = float(
            target_participation_rate
            if target_participation_rate is not None
            else settings.execution.target_participation_rate
        )
        impact_a = float(getattr(settings.market_impact, "coefficient_a", 0.03))

        result = _run_with_execution_settings(
            bars_5m,
            bars_1m,
            ticker=ticker,
            execution_mode=execution_mode,
            target_notional_rub=target_notional_rub,
            signal_threshold=signal_threshold,
            order_size_lots=requested_q,
            dynamic_order_sizing=dynamic,
            target_participation_rate=target_participation,
            max_order_notional_rub=max_order_notional_rub,
            min_order_size_lots=min_order_size_lots,
        )

        pnl_columns = (
            "bar_end_ts_5min",
            "seccode",
            "pnl_mid",
            "pnl_net",
            "implementation_shortfall",
            "optimal_implementation_shortfall",
            "is_reduction_pct",
            "Q_executed",
        )
        pnl_table = _records_frame(result.get("execution_summary", []), pnl_columns)

        ref_price = float(bars_5m["close"].drop_nulls().median() or 0.0) if not bars_5m.is_empty() else 0.0
        q_rows: list[dict[str, Any]] = []
        for q in [float(x) for x in q_grid]:
            q_result = _run_with_execution_settings(
                bars_5m,
                bars_1m,
                ticker=ticker,
                execution_mode=execution_mode,
                target_notional_rub=target_notional_rub,
                signal_threshold=signal_threshold,
                order_size_lots=q,
                dynamic_order_sizing=dynamic,
                target_participation_rate=target_participation,
                max_order_notional_rub=max_order_notional_rub,
                min_order_size_lots=min_order_size_lots,
            )
            metrics = q_result.get("metrics", {})
            shortfall_ratio = float(metrics.get("shortfall_ratio", 0.0) or 0.0)
            q_rows.append(
                {
                    "tested_Q": q,
                    "target_notional": float(target_notional_rub) if target_notional_rub is not None else np.nan,
                    "pnl_mid": float(q_result.get("pnl_mid", 0.0) or 0.0),
                    "pnl_net": float(q_result.get("pnl_net", 0.0) or 0.0),
                    "implementation_shortfall": float(q_result.get("implementation_shortfall", 0.0) or 0.0),
                    "shortfall_ratio": shortfall_ratio,
                    "avg_participation_rate": float(metrics.get("avg_participation_rate", 0.0) or 0.0),
                    "max_participation_rate": float(metrics.get("max_participation_rate", 0.0) or 0.0),
                    "safety_zone": _shortfall_zone(shortfall_ratio),
                    "estimated_aum_proxy_rub": float(q * ref_price) if ref_price > 0 else 0.0,
                    "estimated_optimal_aum_rub": np.nan,
                    "estimate_label": "conservative Q * median price proxy",
                }
            )

        safe_rows = [
            row for row in q_rows
            if row["safety_zone"] in {"SAFE", "ACCEPTABLE"} and np.isfinite(row["pnl_net"])
        ]
        selected_row = max(safe_rows, key=lambda row: row["pnl_net"], default=None)
        estimated_optimal_aum_rub = float(selected_row["estimated_aum_proxy_rub"]) if selected_row else 0.0
        if selected_row is not None:
            selected_row["estimated_optimal_aum_rub"] = estimated_optimal_aum_rub

        aum_table = pd.DataFrame(q_rows)

        formal_validation = validate_formal_output(result)
        validation_ok = all(row.get("ok") for row in formal_validation.values())
        metrics = {
            **result.get("metrics", {}),
            "pnl_mid": float(result.get("pnl_mid", 0.0) or 0.0),
            "pnl_net": float(result.get("pnl_net", 0.0) or 0.0),
            "implementation_shortfall": float(result.get("implementation_shortfall", 0.0) or 0.0),
            "trades_count": int(result.get("total", 0) or 0),
            "winrate": float(result.get("winrate", 0.0) or 0.0),
            "formal_validation_ok": bool(validation_ok),
            "estimated_optimal_aum_rub": estimated_optimal_aum_rub,
            "estimated_optimal_aum_label": "estimate: best SAFE/ACCEPTABLE tested Q by pnl_net times median price",
        }

        params = {
            "seccode": ticker,
            "ticker": ticker,
            "bars": int(bars),
            "execution_mode": execution_mode,
            "order_size_lots": requested_q,
            "Q": requested_q,
            "dynamic_order_sizing": dynamic,
            "target_participation_rate": target_participation,
            "target_notional_rub": target_notional_rub,
            "signal_threshold_delta": float(signal_threshold),
            "impact_coefficient_a": impact_a,
            "max_order_notional_rub": max_order_notional_rub,
            "min_order_size_lots": min_order_size_lots,
        }

        return {
            "candles_5min": _pl_to_pandas(bars_5m),
            "candles_1min": _pl_to_pandas(bars_1m),
            "signal_5min": pd.DataFrame(result.get("signal_5min", [])),
            "backtest_baseline": pd.DataFrame(result.get("backtest_baseline", [])),
            "impact_model": pd.DataFrame(result.get("impact_model", [])),
            "execution_schedule": pd.DataFrame(result.get("execution_schedule", [])),
            "execution_summary": pd.DataFrame(result.get("execution_summary", [])),
            "pnl_table": pnl_table,
            "aum_table": aum_table,
            "params": params,
            "metrics": metrics,
            "formal_validation": formal_validation,
            "raw_result": result,
        }


Pypeline = Pipeline
