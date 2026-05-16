"""
Main Pipeline: M1 → M2 → M3 → M4 → M5 → Optimal AUM
======================================================
Usage:
    python main.py \
        --folder_1min ./is_features_1_min_hackaton \
        --folder_5min ./is_features_5_min_hackaton  \
        --aum 50000000 \
        --method vwap \
        --output ./output
"""

import argparse
import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Add modules dir to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_loader  import DataLoader
from modules.m2_signal    import SignalGenerator
from modules.m3_backtest  import BaselineBacktest
from modules.m4_impact    import ImpactModel
from modules.m5_execution import ExecutionOptimizer
from modules.optimal_aum  import compute_optimal_aum


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(
    folder_1min: str,
    folder_5min: str,
    aum:         float = 50_000_000,
    method:      str   = "vwap",
    delta:       float = 0.1,
    ema_fast:    int   = 5,
    ema_slow:    int   = 20,
    a_fixed:     float = None,
    output_dir:  str   = "./output",
) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ── M1: Load data ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("M1: Loading and quantizing data")
    print("="*60)
    loader = DataLoader(folder_1min, folder_5min).load()
    bars_5 = loader.bars_5min()
    bars_1 = loader.bars_1min()
    adv_df = loader.adv(bars_1)  # use 1-min bars for ADV — 5-min volume may be zero-filled

    print(f"  5-min bars: {len(bars_5):,} rows, {bars_5['seccode'].nunique()} tickers")
    print(f"  1-min bars: {len(bars_1):,} rows")
    print(f"  Date range: {bars_5['timestamp'].min()} → {bars_5['timestamp'].max()}")

    # ── M2: Signal ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("M2: Computing signal")
    print("="*60)
    sig_gen = SignalGenerator(ema_fast=ema_fast, ema_slow=ema_slow, delta=delta)
    signal  = sig_gen.compute(bars_5)
    n_active = signal["value"].notna().sum()
    print(f"  Signal bars: {len(signal):,} total, {n_active:,} active (|val| ≥ δ={delta})")
    print(f"  Signal stats: mean={signal['value'].mean():.4f}, "
          f"std={signal['value'].std():.4f}")

    # ── M3: Baseline backtest ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("M3: Baseline backtest (zero impact)")
    print("="*60)
    bt = BaselineBacktest(delta=delta)
    backtest = bt.run(signal, bars_5)
    stats    = bt.summary(backtest)
    print(f"  Sharpe ratio:     {stats['sharpe_ratio']}")
    print(f"  Hit rate:         {stats['hit_rate']:.2%}")
    print(f"  Total PnL (mid):  {stats['total_pnl_mid']:.6f}")
    print(f"  N bars evaluated: {stats['n_bars']:,}")

    # ── M4: Impact model ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("M4: Calibrating impact model")
    print("="*60)
    impact_model = ImpactModel(a_fixed=a_fixed)
    impact_model.calibrate(bars_1)
    impact_table = impact_model.compute(bars_1)
    a_summary = impact_model.summary()
    print(f"  Calibrated {len(a_summary)} tickers")
    print(f"  `a` range: [{a_summary['a'].min():.4f}, {a_summary['a'].max():.4f}]")
    print(f"  `a` mean:  {a_summary['a'].mean():.4f}")

    # ── M5: Execution optimization ────────────────────────────────────────
    print("\n" + "="*60)
    print(f"M5: Execution optimization (method={method})")
    print("="*60)
    exec_opt = ExecutionOptimizer(aum=aum, method=method)
    schedule, exec_summary = exec_opt.run(signal, bars_1, impact_table, backtest)

    if not exec_summary.empty:
        is_comparison = exec_opt.compare_is(exec_summary)
        total_is  = exec_summary["implementation_shortfall"].sum()
        total_net = exec_summary["pnl_net"].sum()
        print(f"  Total IS:         {total_is:,.2f}")
        print(f"  Total net PnL:    {total_net:,.2f}")
        print(f"  IS reduction:     {is_comparison.get('IS_reduction', 0):.2%}")
        sharpe_net = _sharpe(exec_summary.groupby("bar_end_ts_5min")["pnl_net"].sum())
        print(f"  Sharpe (net PnL): {sharpe_net:.4f}")
    else:
        is_comparison = {}
        print("  No execution records generated (check signal & 1-min data overlap).")

    # ── Optimal AUM ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Optimal AUM estimation")
    print("="*60)
    opt_aum = compute_optimal_aum(backtest, adv_df, a_summary, bars_1)
    total_x_star = opt_aum["X_star"].sum()
    print(f"  Total X* (portfolio): {total_x_star:,.0f} RUB")
    print(f"  Top 5 tickers by X*:")
    print(opt_aum.head(5)[["seccode","alpha","adv","a","X_star"]].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Saving outputs")
    print("="*60)
    _save(signal,       os.path.join(output_dir, "signal_5min.parquet"))
    _save(backtest,     os.path.join(output_dir, "backtest_baseline.parquet"))
    _save(impact_table, os.path.join(output_dir, "impact_model.parquet"))
    _save(a_summary,    os.path.join(output_dir, "impact_calibration.parquet"))
    _save(schedule,     os.path.join(output_dir, "execution_schedule.parquet"))
    _save(exec_summary, os.path.join(output_dir, "execution_summary.parquet"))
    _save(opt_aum,      os.path.join(output_dir, "optimal_aum.parquet"))

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs in: {os.path.abspath(output_dir)}")

    # ── Consolidated report ───────────────────────────────────────────────
    report = {
        "M2_signal": {
            "n_total_bars":  len(signal),
            "n_active_bars": int(n_active),
        },
        "M3_baseline": stats,
        "M4_impact": {
            "a_mean":  round(float(a_summary["a"].mean()), 4),
            "a_min":   round(float(a_summary["a"].min()), 4),
            "a_max":   round(float(a_summary["a"].max()), 4),
        },
        "M5_execution": {
            "total_IS":       round(total_is, 2) if not exec_summary.empty else 0,
            "total_pnl_net":  round(total_net, 2) if not exec_summary.empty else 0,
            **is_comparison,
        },
        "optimal_aum": {
            "total_X_star_RUB": round(total_x_star, 0),
        },
        "runtime_seconds": round(elapsed, 1),
    }

    with open(os.path.join(output_dir, "pipeline_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, path: str):
    if df is not None and not df.empty:
        df.to_parquet(path, index=False)
        print(f"  Saved: {os.path.basename(path)} ({len(df):,} rows)")
    else:
        print(f"  Skipped (empty): {os.path.basename(path)}")


def _sharpe(pnl: pd.Series) -> float:
    if pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252 * 51))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Trading Signal Pipeline")
    parser.add_argument("--folder_1min", default="./data/is_features_1_min_hackaton",
                        help="Path to is_features_1_min_hackaton folder")
    parser.add_argument("--folder_5min", default="./data/is_features_5_min_hackaton",
                        help="Path to is_features_5_min_hackaton folder")
    parser.add_argument("--aum",    type=float, default=50_000_000)
    parser.add_argument("--method", choices=["twap","vwap","numeric"], default="vwap")
    parser.add_argument("--delta",  type=float, default=0.1)
    parser.add_argument("--ema_fast", type=int, default=5)
    parser.add_argument("--ema_slow", type=int, default=20)
    parser.add_argument("--a_fixed",  type=float, default=None,
                        help="Fixed impact coef (skip calibration)")
    parser.add_argument("--output",   default="./output")
    args = parser.parse_args()

    report = run_pipeline(
        folder_1min = args.folder_1min,
        folder_5min = args.folder_5min,
        aum         = args.aum,
        method      = args.method,
        delta       = args.delta,
        ema_fast    = args.ema_fast,
        ema_slow    = args.ema_slow,
        a_fixed     = args.a_fixed,
        output_dir  = args.output,
    )
    print("\n── Pipeline Report ──")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
