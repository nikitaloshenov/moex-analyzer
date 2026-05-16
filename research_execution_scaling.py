from __future__ import annotations

import argparse
from statistics import mean

import polars as pl

from app.core.config import settings
from app.services.backtest import run_pattern_backtest
from app.services.ingestion import load_feature_pair


SCALING_HEADER = (
    f"{'Q_req':>8} | {'avg_Q':>8} | {'max_Q':>8} | {'status':<10} | "
    f"{'trades':>6} | {'pnl_mid':>10} | {'pnl_net':>10} | {'IS':>10} | "
    f"{'sf':>5} | {'IS_red':>6} | {'avg_part':>8} | {'max_part':>8} | zone"
)


def _shortfall_zone(shortfall_ratio: float) -> str:
    if shortfall_ratio < 0.5:
        return "SAFE"
    if shortfall_ratio < 1.0:
        return "ACCEPTABLE"
    if shortfall_ratio < 2.0:
        return "DANGEROUS"
    return "BROKEN"


def _print_scaling_header() -> None:
    print("=" * len(SCALING_HEADER))
    print(SCALING_HEADER)
    print("=" * len(SCALING_HEADER))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Q scaling test")
    parser.add_argument("--data-root", default="insample_data")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--bars", type=int, default=300)
    parser.add_argument("--q-grid", default="100,500,1000,5000,10000,25000")
    parser.add_argument("--execution-mode", choices=("twap", "optimal"), default="twap")
    parser.add_argument("--dynamic-order-sizing", dest="dynamic_order_sizing", action="store_true", default=None)
    parser.add_argument("--no-dynamic-order-sizing", dest="dynamic_order_sizing", action="store_false")
    parser.add_argument("--target-participation-rate", type=float, default=None)
    parser.add_argument("--max-order-notional-rub", type=float, default=None)
    parser.add_argument("--min-order-size-lots", type=float, default=None)
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    q_grid = [float(x.strip()) for x in args.q_grid.split(",") if x.strip()]

    print("=" * 80)
    print("FORMAL Q SCALING TEST")
    print(f"ticker: {ticker}")
    print(f"bars: {args.bars}")
    print(f"q_grid: {q_grid}")
    print(f"execution_mode: {args.execution_mode}")
    print(f"dynamic_order_sizing: {args.dynamic_order_sizing if args.dynamic_order_sizing is not None else settings.execution.dynamic_order_sizing}")
    print(f"target_participation_rate: {args.target_participation_rate if args.target_participation_rate is not None else settings.execution.target_participation_rate}")
    print(f"max_order_notional_rub: {args.max_order_notional_rub if args.max_order_notional_rub is not None else settings.execution.max_order_notional_rub}")
    print("=" * 80)

    bars_5m, bars_1m = load_feature_pair(args.data_root, ticker)

    if args.bars > 0:
        bars_5m = bars_5m.head(args.bars)
        if not bars_5m.is_empty():
            last_ts = bars_5m["begin"][-1]
            bars_1m = bars_1m.filter(pl.col("begin") <= last_ts)

    print(f"loaded bars_5m: {bars_5m.height}")
    print(f"loaded bars_1m: {bars_1m.height}")
    _print_scaling_header()

    original_q = float(settings.execution.order_size_lots)
    original_dynamic_order_sizing = bool(settings.execution.dynamic_order_sizing)
    original_target_participation_rate = float(settings.execution.target_participation_rate)
    original_max_order_notional_rub = settings.execution.max_order_notional_rub
    original_min_order_size_lots = float(settings.execution.min_order_size_lots)
    rows = []

    try:
        if args.dynamic_order_sizing is not None:
            settings.execution.dynamic_order_sizing = args.dynamic_order_sizing
        if args.target_participation_rate is not None:
            settings.execution.target_participation_rate = args.target_participation_rate
        if args.max_order_notional_rub is not None:
            settings.execution.max_order_notional_rub = args.max_order_notional_rub
        if args.min_order_size_lots is not None:
            settings.execution.min_order_size_lots = args.min_order_size_lots

        for q in q_grid:
            settings.execution.order_size_lots = q

            result = run_pattern_backtest(
                bars_5m,
                bars_1m,
                ticker=ticker,
                execution_mode=args.execution_mode,
            )

            metrics = result.get("metrics", {})
            row = {
                "requested_Q": q,
                "status": result.get("status"),
                "trades": int(result.get("total", 0) or 0),
                "pnl_mid": float(result.get("pnl_mid", 0.0) or 0.0),
                "pnl_net": float(result.get("pnl_net", 0.0) or 0.0),
                "IS": float(result.get("implementation_shortfall", 0.0) or 0.0),
                "shortfall_ratio": float(metrics.get("shortfall_ratio", 0.0) or 0.0),
                "avg_Q_executed": float(metrics.get("avg_Q_executed", 0.0) or 0.0),
                "max_Q_executed": float(metrics.get("max_Q_executed", 0.0) or 0.0),
                "avg_part": float(metrics.get("avg_participation_rate", 0.0) or 0.0),
                "max_part": float(metrics.get("max_participation_rate", 0.0) or 0.0),
                "is_reduction_pct": float(metrics.get("is_reduction_pct", 0.0) or 0.0),
                "sharpe": float(metrics.get("sharpe_pnl_net", 0.0) or 0.0),
            }
            row["zone"] = _shortfall_zone(row["shortfall_ratio"])
            rows.append(row)

            print(
                f"{row['requested_Q']:>8.0f} | "
                f"{row['avg_Q_executed']:>8.0f} | "
                f"{row['max_Q_executed']:>8.0f} | "
                f"{row['status']:<10} | "
                f"{row['trades']:>6} | "
                f"{row['pnl_mid']:>10.0f} | "
                f"{row['pnl_net']:>10.0f} | "
                f"{row['IS']:>10.0f} | "
                f"{row['shortfall_ratio']:>5.3f} | "
                f"{row['is_reduction_pct']:>6.2f} | "
                f"{row['avg_part']:>8.4f} | "
                f"{row['max_part']:>8.4f} | "
                f"{row['zone']}"
            )

    finally:
        settings.execution.order_size_lots = original_q
        settings.execution.dynamic_order_sizing = original_dynamic_order_sizing
        settings.execution.target_participation_rate = original_target_participation_rate
        settings.execution.max_order_notional_rub = original_max_order_notional_rub
        settings.execution.min_order_size_lots = original_min_order_size_lots

    print("=" * len(SCALING_HEADER))

    if args.save_csv:
        pl.DataFrame(rows).write_csv(args.save_csv)
        print(f"saved_csv: {args.save_csv}")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["pnl_net"])
        print("BEST BY pnl_net:")
        print(
            f"requested_Q={best['requested_Q']:.0f}, "
            f"avg_Q_executed={best['avg_Q_executed']:.2f}, "
            f"max_Q_executed={best['max_Q_executed']:.2f}, "
            f"pnl_net={best['pnl_net']:.2f}, "
            f"shortfall_ratio={best['shortfall_ratio']:.3f}, "
            f"max_participation={best['max_part']:.4f}, "
            f"is_reduction_pct={best['is_reduction_pct']:.2f}"
        )

        print("TOTAL:")
        print(f"tested_q: {len(rows)}")
        print(f"ok_q: {len(ok)}")
        print(f"avg_pnl_net: {mean([r['pnl_net'] for r in ok]):.2f}")
    else:
        print("NO OK RESULTS")

    print("=" * 80)


if __name__ == "__main__":
    main()
