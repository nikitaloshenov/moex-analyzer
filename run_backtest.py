from __future__ import annotations

import argparse

from app.core.config import settings
from app.services.backtest import run_pattern_backtest, validate_formal_output
from app.services.ingestion import discover_feature_tickers, load_feature_pair


COMPACT_HEADER = (
    f"{'ticker':<6} | {'mode':<7} | {'trades':>6} | {'pnl_mid':>9} | "
    f"{'pnl_net':>9} | {'IS':>9} | {'sf':>5} | {'IS_red':>6} | "
    f"{'avg_part':>8} | {'max_part':>8} | {'avg_Q':>6} | {'max_Q':>6} | zone"
)


def _shortfall_zone(shortfall_ratio: float) -> str:
    if shortfall_ratio < 0.5:
        return "SAFE"
    if shortfall_ratio < 1.0:
        return "ACCEPTABLE"
    if shortfall_ratio < 2.0:
        return "DANGEROUS"
    return "BROKEN"


def _print_compact_header() -> None:
    print("=" * len(COMPACT_HEADER))
    print(COMPACT_HEADER)
    print("=" * len(COMPACT_HEADER))


def _print_formal_validation(result: dict) -> bool:
    report = validate_formal_output(result)
    print("formal_output_validation:")
    ok_all = True
    for table, row in report.items():
        ok = bool(row["ok"])
        ok_all = ok_all and ok
        status = "OK" if ok else "FAIL"
        missing = ",".join(row["missing"]) if row["missing"] else "-"
        print(f"  {table}: {status} rows={row['rows']} missing={missing}")
    return ok_all


def _run_single(
    *,
    data_root: str,
    ticker: str,
    bars: int,
    execution_mode: str,
    target_notional_rub: float | None,
    order_size_lots: float | None,
    dynamic_order_sizing: bool | None,
    target_participation_rate: float | None,
    max_order_notional_rub: float | None,
    min_order_size_lots: float | None,
    compact: bool,
) -> None:
    bars_5m, bars_1m = load_feature_pair(data_root, ticker)

    if bars > 0:
        bars_5m = bars_5m.head(bars)

        if not bars_5m.is_empty():
            last_ts = bars_5m["begin"][-1]
            bars_1m = bars_1m.filter(bars_1m["begin"] <= last_ts)

    original_execution = {
        "order_size_lots": settings.execution.order_size_lots,
        "dynamic_order_sizing": settings.execution.dynamic_order_sizing,
        "target_participation_rate": settings.execution.target_participation_rate,
        "max_order_notional_rub": settings.execution.max_order_notional_rub,
        "min_order_size_lots": settings.execution.min_order_size_lots,
    }

    try:
        if order_size_lots is not None:
            settings.execution.order_size_lots = order_size_lots
        if dynamic_order_sizing is not None:
            settings.execution.dynamic_order_sizing = dynamic_order_sizing
        if target_participation_rate is not None:
            settings.execution.target_participation_rate = target_participation_rate
        if max_order_notional_rub is not None:
            settings.execution.max_order_notional_rub = max_order_notional_rub
        if min_order_size_lots is not None:
            settings.execution.min_order_size_lots = min_order_size_lots

        result = run_pattern_backtest(
            bars_5m,
            bars_1m,
            ticker=ticker,
            execution_mode=execution_mode,
            target_notional_rub=target_notional_rub,
        )
    finally:
        for key, value in original_execution.items():
            setattr(settings.execution, key, value)

    metrics = result.get("metrics", {})

    if compact:
        trades_count = int(result.get("total", 0) or 0)
        if trades_count == 0:
            print(f"{ticker:<6} | {execution_mode:<7} | NO_TRADES")
            return

        shortfall_ratio = float(metrics.get("shortfall_ratio", 0.0) or 0.0)
        print(
            f"{ticker:<6} | {execution_mode:<7} | "
            f"{trades_count:>6} | "
            f"{float(result.get('pnl_mid', 0.0) or 0.0):>9.0f} | "
            f"{float(result.get('pnl_net', 0.0) or 0.0):>9.0f} | "
            f"{float(result.get('implementation_shortfall', 0.0) or 0.0):>9.0f} | "
            f"{shortfall_ratio:>5.3f} | "
            f"{float(metrics.get('is_reduction_pct', 0.0) or 0.0):>6.2f} | "
            f"{float(metrics.get('avg_participation_rate', 0.0) or 0.0):>8.4f} | "
            f"{float(metrics.get('max_participation_rate', 0.0) or 0.0):>8.4f} | "
            f"{float(metrics.get('avg_Q_executed', 0.0) or 0.0):>6.0f} | "
            f"{float(metrics.get('max_Q_executed', 0.0) or 0.0):>6.0f} | "
            f"{_shortfall_zone(shortfall_ratio)}"
        )
        return

    print("=" * 100)
    print(f"TICKER: {ticker}")
    print("=" * 100)

    print(f"status: {result.get('status')}")
    print(f"execution_mode: {execution_mode}")
    requested_order_size = order_size_lots if order_size_lots is not None else original_execution["order_size_lots"]
    print(f"target_notional_rub: {target_notional_rub}")
    print(f"order_size_lots_requested: {requested_order_size:.2f}")
    print(f"dynamic_order_sizing: {dynamic_order_sizing if dynamic_order_sizing is not None else original_execution['dynamic_order_sizing']}")
    print(f"target_participation_rate: {target_participation_rate if target_participation_rate is not None else original_execution['target_participation_rate']:.4f}")
    max_notional_out = max_order_notional_rub if max_order_notional_rub is not None else original_execution["max_order_notional_rub"]
    print(f"max_order_notional_rub: {max_notional_out}")
    print(f"trades_count: {result.get('total')}")
    print(f"pnl_mid: {result.get('pnl_mid'):.2f}")
    print(f"pnl_net: {result.get('pnl_net'):.2f}")
    print(f"implementation_shortfall: {result.get('implementation_shortfall'):.2f}")
    print(f"shortfall_ratio: {metrics.get('shortfall_ratio'):.3f}")
    print(f"winrate: {result.get('winrate'):.2f}")
    print(f"hit_rate_mid: {metrics.get('hit_rate_mid'):.2f}")
    print(f"sharpe_pnl_net: {metrics.get('sharpe_pnl_net'):.3f}")
    print(f"is_reduction_pct: {metrics.get('is_reduction_pct'):.2f}")
    print(f"avg_Q_executed: {metrics.get('avg_Q_executed'):.2f}")
    print(f"max_Q_executed: {metrics.get('max_Q_executed'):.2f}")
    print(f"approx_turnover_rub: {metrics.get('approx_turnover_rub'):.2f}")
    print(f"pnl_net_pct_of_turnover: {metrics.get('pnl_net_pct_of_turnover'):.4f}")
    print(f"implementation_shortfall_pct_of_turnover: {metrics.get('implementation_shortfall_pct_of_turnover'):.4f}")
    print(f"avg_participation_rate: {metrics.get('avg_participation_rate'):.4f}")
    print(f"max_participation_rate: {metrics.get('max_participation_rate'):.4f}")
    print("vwap_fill_note: achieved volume-weighted fill price of selected schedule; not VWAP execution mode")
    _print_formal_validation(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean FORMAL_TASK M1-M5 backtest runner.")

    parser.add_argument("--data-root", default="insample_data")

    parser.add_argument("--ticker", default="SBER")

    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers, e.g. SBER,GAZP,LKOH",
    )
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--max-tickers", type=int, default=None)

    parser.add_argument("--bars", type=int, default=1000)

    parser.add_argument(
        "--execution-mode",
        choices=("twap", "optimal"),
        default="twap",
    )

    parser.add_argument("--target-notional-rub", type=float, default=None)
    parser.add_argument("--order-size-lots", type=float, default=None)
    parser.add_argument("--dynamic-order-sizing", dest="dynamic_order_sizing", action="store_true", default=None)
    parser.add_argument("--no-dynamic-order-sizing", dest="dynamic_order_sizing", action="store_false")
    parser.add_argument("--target-participation-rate", type=float, default=None)
    parser.add_argument("--max-order-notional-rub", type=float, default=None)
    parser.add_argument("--min-order-size-lots", type=float, default=None)
    parser.add_argument("--compact", action="store_true")

    args = parser.parse_args()

    if args.all_tickers:
        tickers = discover_feature_tickers(args.data_root)
        if args.max_tickers is not None and args.max_tickers > 0:
            tickers = tickers[: args.max_tickers]
    elif args.tickers:
        tickers = [
            x.strip().upper()
            for x in args.tickers.split(",")
            if x.strip()
        ]
    else:
        tickers = [args.ticker.upper()]

    if args.compact:
        _print_compact_header()

    for ticker in tickers:
        try:
            _run_single(
                data_root=args.data_root,
                ticker=ticker,
                bars=args.bars,
                execution_mode=args.execution_mode,
                target_notional_rub=args.target_notional_rub,
                order_size_lots=args.order_size_lots,
                dynamic_order_sizing=args.dynamic_order_sizing,
                target_participation_rate=args.target_participation_rate,
                max_order_notional_rub=args.max_order_notional_rub,
                min_order_size_lots=args.min_order_size_lots,
                compact=args.compact,
            )
        except Exception as e:
            if args.compact:
                print(f"{ticker} | FAILED | {e}")
            else:
                print("=" * 100)
                print(f"TICKER FAILED: {ticker}")
                print(e)


if __name__ == "__main__":
    main()  
