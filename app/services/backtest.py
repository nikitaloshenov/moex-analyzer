"""
Высокопроизводительный walk-forward бэктест (V2).

Polars-хранилище, очистка, режим рынка (regime), симуляция TP/SL,
market impact, метрики, список сделок для JSON/графиков.

Связь: PatternAnalyzer.analyze_all(pl-слайс); параметры: app.core.config.settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from app.core.config import settings
from app.services.analyzer import PatternAnalyzer
from app.services.regime import classify_regime, regime_allows_signal

# -----------------------------------------------------------------------------
# Константы и legacy-модель
# -----------------------------------------------------------------------------

BACKTEST_VERSION = "2.2"


@dataclass(frozen=True)
class MarketImpactConfig:
    """Микроструктурное проскальзывание (если position_notional_rub=None)."""

    participation: float = 0.05
    impact_k: float = 0.0015


def _slippage_abs(price: float, bar_volume: float, cfg: MarketImpactConfig) -> float:
    if price <= 0 or bar_volume <= 0:
        return 0.0
    return float(price) * cfg.impact_k * cfg.participation


def _execution_slip_multiplier(mode: str) -> float:
    if mode == "twap":
        return 0.72
    if mode == "vwap":
        return 0.82
    return 1.0


def _turnover_slip_fraction(notional_rub: float, avg_turnover_rub: float, exec_mult: float) -> float:
    mi = settings.market_impact
    if notional_rub <= 0 or avg_turnover_rub <= 0:
        return 0.0
    ratio_pct = 100.0 * notional_rub / avg_turnover_rub
    if ratio_pct <= mi.turnover_threshold_pct:
        return 0.0
    span = max(50.0 - mi.turnover_threshold_pct, 1e-6)
    t = min(1.0, (ratio_pct - mi.turnover_threshold_pct) / span)
    slip_pct = mi.slip_min_pct + t * (mi.slip_max_pct - mi.slip_min_pct)
    return (slip_pct / 100.0) * exec_mult


# -----------------------------------------------------------------------------
# Очистка и оборот
# -----------------------------------------------------------------------------


def _apply_data_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    dc = settings.data_cleaning
    if not dc.enabled:
        return df
    w = dc.volume_spike_window
    z = dc.volume_spike_zscore
    v = pl.col("volume")
    mu = v.rolling_mean(window_size=w, min_periods=max(3, w // 4))
    sd = v.rolling_std(window_size=w, min_periods=max(3, w // 4)).fill_null(1e-9).clip(lower_bound=1e-9)
    hi = mu + z * sd
    lo = (mu - z * sd).clip(lower_bound=0.0)
    v_clip = pl.min_horizontal(pl.max_horizontal(v, lo), hi)
    return df.with_columns(v_clip.alias("volume"))


def _with_avg_turnover_column(df: pl.DataFrame) -> pl.DataFrame:
    w = settings.market_impact.avg_turnover_lookback_days
    tov = pl.col("close") * pl.col("volume")
    return df.with_columns(
        tov.rolling_mean(window_size=w, min_periods=max(3, w // 3)).alias("_avg_turnover_rub"),
    )


def _prepare_pl(candles_df) -> pl.DataFrame:
    if isinstance(candles_df, pl.DataFrame):
        return candles_df.clone()
    return pl.from_pandas(candles_df.reset_index(drop=True))


def _find_start_row(df: pl.DataFrame, start_date: str) -> int | None:
    if df.is_empty():
        return None
    d0 = start_date[:10]
    begin = df["begin"]
    dt = begin.dtype
    if dt in (pl.Utf8, pl.String):
        key = begin.str.slice(0, 10)
    elif str(dt).startswith("Datetime"):
        key = begin.dt.strftime("%Y-%m-%d")
    else:
        key = begin.cast(pl.Utf8).str.slice(0, 10)
    w = df.with_row_index().filter(key >= d0).head(1)
    if w.is_empty():
        return None
    return int(w["index"].item())


def _bar_ts(beg_col: pl.Series, idx: int) -> str:
    try:
        return str(beg_col[idx])[:32]
    except Exception:
        return str(idx)


# -----------------------------------------------------------------------------
# Симуляция сделки вперёд
# -----------------------------------------------------------------------------


def _forward_trade_pnl(
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    i: int,
    entry: float,
    tp: float,
    sl: float,
    is_buy: bool,
    slip_frac: float,
    legacy: MarketImpactConfig | None,
) -> tuple[float, float, float, int, Literal["tp", "sl"]] | None:
    """pnl, slip_in, slip_out, exit_bar_index, exit_kind."""
    n = high.shape[0]
    if i + 1 >= n:
        return None

    v_entry = float(vol[i])
    leg_in = _slippage_abs(entry, v_entry, legacy) if legacy else 0.0
    slip_in = float(entry) * slip_frac + leg_in

    if is_buy:
        adj_entry = entry + slip_in
        for j in range(i + 1, n):
            vj = float(vol[j])
            if high[j] >= tp:
                leg_out = _slippage_abs(float(tp), vj, legacy) if legacy else 0.0
                slip_out = float(tp) * slip_frac + leg_out
                exit_px = float(tp) - slip_out
                return exit_px - adj_entry, slip_in, slip_out, j, "tp"
            if low[j] <= sl:
                leg_out = _slippage_abs(float(sl), vj, legacy) if legacy else 0.0
                slip_out = float(sl) * slip_frac + leg_out
                exit_px = float(sl) - slip_out
                return exit_px - adj_entry, slip_in, slip_out, j, "sl"
    else:
        adj_entry = entry - slip_in
        for j in range(i + 1, n):
            vj = float(vol[j])
            if low[j] <= tp:
                leg_out = _slippage_abs(float(tp), vj, legacy) if legacy else 0.0
                slip_out = float(tp) * slip_frac + leg_out
                exit_px = float(tp) + slip_out
                return adj_entry - exit_px, slip_in, slip_out, j, "tp"
            if high[j] >= sl:
                leg_out = _slippage_abs(float(sl), vj, legacy) if legacy else 0.0
                slip_out = float(sl) * slip_frac + leg_out
                exit_px = float(sl) + slip_out
                return adj_entry - exit_px, slip_in, slip_out, j, "sl"
    return None


# -----------------------------------------------------------------------------
# Метрики
# -----------------------------------------------------------------------------


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    base = np.where(np.abs(peak) < 1e-12, 1.0, peak)
    dd = (equity - peak) / base * 100.0
    return float(dd.min())


def _sharpe_sortino(pnls: list[float], entries: list[float]) -> tuple[float, float]:
    bm = settings.backtest_metrics
    rf_p = bm.risk_free_rate_annual / float(bm.periods_per_year)
    rets = np.array([p / abs(e) for p, e in zip(pnls, entries) if abs(e) > 1e-12], dtype=np.float64)
    if rets.size < 2:
        return 0.0, 0.0
    xs = rets - rf_p
    std = float(rets.std(ddof=1))
    if std < 1e-12:
        return 0.0, 0.0
    ann = np.sqrt(float(bm.periods_per_year))
    sharpe = float(xs.mean() / std * ann)
    downside = rets[rets < 0]
    if downside.size > 1:
        dstd = float(downside.std(ddof=1))
        sortino = float(xs.mean() / dstd * ann) if dstd > 1e-12 else sharpe
    else:
        sortino = sharpe
    return sharpe, sortino


# -----------------------------------------------------------------------------
# JSON сделок
# -----------------------------------------------------------------------------


def trades_to_json(trades: list[dict[str, Any]]) -> str:
    """Сериализация списка сделок для сохранения в файл или отдачи в API."""
    return json.dumps(trades, ensure_ascii=False, indent=2, default=str)


# -----------------------------------------------------------------------------
# Публичный бэктест
# -----------------------------------------------------------------------------


def run_pattern_backtest(
    candles,
    start_date: str,
    ticker: str = "DEFAULT",
    *,
    position_notional_rub: float | None = None,
    execution_mode: Literal["immediate", "twap", "vwap"] | None = None,
    legacy_impact: MarketImpactConfig | None = None,
    single_position: bool = False,
    min_rows: int = 50,
) -> dict[str, Any]:
    """
    single_position=True: после выхода из сделки следующий сигнал не раньше бара exit+1.
    """
    legacy_only = position_notional_rub is None
    legacy = legacy_impact if legacy_impact is not None else (MarketImpactConfig() if legacy_only else None)

    mode = execution_mode or settings.execution.default_mode
    exec_mult = _execution_slip_multiplier(mode)

    empty = {"ticker": ticker, "status": "no_data", "version": BACKTEST_VERSION, "trades": []}

    if candles is None:
        return empty

    df = _prepare_pl(candles)
    if df.height < min_rows:
        return empty

    df = _apply_data_cleaning(df)
    df = _with_avg_turnover_column(df)

    start_row = _find_start_row(df, start_date)
    if start_row is None:
        return empty

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df["volume"].to_numpy().astype(np.float64, copy=False)
    avg_tov = df["_avg_turnover_rub"].to_numpy().astype(np.float64, copy=False)
    n = df.height

    df_ohlc = df.drop("_avg_turnover_rub")
    beg_series = df_ohlc["begin"]

    trade_pnls: list[float] = []
    trade_entries: list[float] = []
    slip_sum_pts: list[float] = []
    trades: list[dict[str, Any]] = []

    i = start_row
    while i < n:
        sub_pl = df_ohlc.slice(0, i + 1)
        regime = classify_regime(sub_pl)

        res = PatternAnalyzer.analyze_all(sub_pl, ticker=ticker)
        sig = res.get("signal", "HOLD")

        if settings.regime.enabled and not regime_allows_signal(sig, regime):
            i += 1
            continue

        if sig not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
            i += 1
            continue

        tp, sl = res.get("take_profit"), res.get("stop_loss")
        if tp is None or sl is None:
            i += 1
            continue

        entry = float(res["price"])
        is_buy = "BUY" in sig

        if legacy_only:
            slip_frac = 0.0
            leg = legacy
        else:
            at = float(avg_tov[i])
            if not np.isfinite(at) or at <= 0:
                slip_frac = 0.0
            else:
                slip_frac = _turnover_slip_fraction(float(position_notional_rub or 0.0), at, exec_mult)
            leg = legacy_impact

        out = _forward_trade_pnl(high, low, vol, i, entry, float(tp), float(sl), is_buy, slip_frac, leg)
        if out is None:
            i += 1
            continue

        pnl, s_in, s_out, j_exit, reason = out
        trade_pnls.append(pnl)
        trade_entries.append(abs(entry))
        slip_sum_pts.append(s_in + s_out)

        trades.append(
            {
                "ticker": ticker,
                "entry_bar": i,
                "exit_bar": j_exit,
                "entry_date": _bar_ts(beg_series, i),
                "exit_date": _bar_ts(beg_series, j_exit),
                "side": "long" if is_buy else "short",
                "signal": sig,
                "regime": regime,
                "entry_price": round(entry, 6),
                "take_profit": float(tp),
                "stop_loss": float(sl),
                "exit_kind": reason,
                "pnl_points": round(pnl, 6),
                "slippage_points": round(s_in + s_out, 8),
            }
        )

        if single_position:
            i = j_exit + 1
        else:
            i += 1

    total_trades = len(trade_pnls)
    wins = [t for t in trade_pnls if t > 0]
    losses = [t for t in trade_pnls if t < 0]

    winrate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_profit = float(sum(trade_pnls)) if trade_pnls else 0.0
    gross_profit = float(sum(wins)) if wins else 0.0
    gross_loss = abs(float(sum(losses))) if losses else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = gross_profit if gross_profit > 0 else 0.0

    eq = np.cumsum(np.array([0.0] + trade_pnls, dtype=np.float64))
    max_dd_pct = _max_drawdown_pct(eq)
    sharpe, sortino = _sharpe_sortino(trade_pnls, trade_entries)
    avg_slip_pct = (
        float(np.mean([100.0 * s / e for s, e in zip(slip_sum_pts, trade_entries) if e > 1e-12]))
        if slip_sum_pts
        else 0.0
    )

    impact_report = {
        "mode": "legacy_participation" if legacy_only else "turnover_plus_optional_legacy",
        "position_notional_rub": position_notional_rub,
        "execution_mode": mode,
        "legacy_participation": legacy.participation if legacy else None,
        "legacy_impact_k": legacy.impact_k if legacy else None,
        "turnover_threshold_pct": settings.market_impact.turnover_threshold_pct,
        "slip_band_pct": [settings.market_impact.slip_min_pct, settings.market_impact.slip_max_pct],
        "single_position": single_position,
    }

    return {
        "ticker": ticker,
        "winrate": winrate,
        "total": total_trades,
        "profit": total_profit,
        "pf": profit_factor,
        "status": "ok",
        "version": BACKTEST_VERSION,
        "impact": impact_report,
        "metrics": {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd_pct,
            "avg_slippage_pct_of_entry": avg_slip_pct,
        },
        "trades": trades,
        "trades_json": trades_to_json(trades),
    }


# -----------------------------------------------------------------------------
# Публичные хелперы (UI)
# -----------------------------------------------------------------------------


def get_execution_slip_multiplier(mode: str) -> float:
    return _execution_slip_multiplier(mode)


def get_turnover_slip_fraction(
    notional_rub: float,
    avg_turnover_rub: float,
    execution_mode: str | None = None,
) -> float:
    mode = execution_mode or settings.execution.default_mode
    return _turnover_slip_fraction(
        notional_rub,
        avg_turnover_rub,
        _execution_slip_multiplier(mode),
    )
