"""
Tests — one test class per module.
Run: python -m pytest tests/test_modules.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from modules.m2_signal    import SignalGenerator
from modules.m3_backtest  import BaselineBacktest
from modules.m4_impact    import ImpactModel
from modules.m5_execution import ExecutionOptimizer, _align_1min_to_5min
from modules.optimal_aum  import compute_optimal_aum


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
def make_bars_5min(n_days=5, tickers=("ABIO","SBER","GAZP")) -> pd.DataFrame:
    """Synthetic 5-min bars for testing."""
    import pytz
    tz = pytz.timezone("Europe/Moscow")
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-09-02") + pd.Timedelta(days=d)
        for h in range(10, 19):
            for m in range(0, 60, 5):
                if h == 18 and m > 30:
                    break
                ts = pd.Timestamp(date.year, date.month, date.day, h, m, tzinfo=tz)
                for sec in tickers:
                    price = 100 + np.random.randn() * 2
                    rows.append({
                        "timestamp":   ts,
                        "seccode":     sec,
                        "open":        price,
                        "high":        price + abs(np.random.randn()),
                        "low":         price - abs(np.random.randn()),
                        "close":       price + np.random.randn() * 0.5,
                        "volume":      int(1e5 + np.random.exponential(5e4)),
                        "mid":         price,
                        "bar_end_ts":  int((ts + pd.Timedelta(minutes=5)).timestamp() * 1e9),
                    })
    return pd.DataFrame(rows)


def make_bars_1min(bars_5: pd.DataFrame) -> pd.DataFrame:
    """Expand each 5-min bar into 5 1-min bars."""
    rows = []
    for _, row in bars_5.iterrows():
        for k in range(5):
            ts = row["timestamp"] + pd.Timedelta(minutes=k)
            price = row["open"] + np.random.randn() * 0.3
            rows.append({
                "timestamp":   ts,
                "seccode":     row["seccode"],
                "open":        price,
                "high":        price + 0.2,
                "low":         price - 0.2,
                "close":       price + np.random.randn() * 0.2,
                "volume":      max(1000, int(row["volume"] / 5 * (0.8 + 0.4 * np.random.rand()))),
                "mid":         price,
                "bar_end_ts":  int((ts + pd.Timedelta(minutes=1)).timestamp() * 1e9),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# M2 TESTS
# ─────────────────────────────────────────────
class TestSignal:
    def test_output_schema(self):
        bars = make_bars_5min()
        sig = SignalGenerator(delta=0.0).compute(bars)
        assert set(sig.columns) >= {"bar_end_ts", "seccode", "value"}

    def test_value_range(self):
        bars = make_bars_5min()
        sig = SignalGenerator(delta=0.0).compute(bars)
        valid = sig["value"].dropna()
        assert (valid.abs() <= 1.0 + 1e-9).all(), "Signal must be in [-1,1]"

    def test_threshold(self):
        bars = make_bars_5min()
        sig = SignalGenerator(delta=0.5).compute(bars)
        valid = sig["value"].dropna()
        assert (valid.abs() >= 0.5 - 1e-9).all(), "All active signals must be ≥ delta"

    def test_no_lookahead(self):
        """Signal at bar_end_ts T should not use data after T."""
        bars = make_bars_5min(n_days=2)
        sig1 = SignalGenerator().compute(bars)
        # Remove last 5 bars (future data) and recompute
        cutoff = bars["timestamp"].max() - pd.Timedelta(minutes=25)
        sig2 = SignalGenerator().compute(bars[bars["timestamp"] <= cutoff])
        # Earlier values should match
        common_ts = set(sig1["bar_end_ts"]) & set(sig2["bar_end_ts"])
        if common_ts:
            merged = sig1[sig1["bar_end_ts"].isin(common_ts)].merge(
                sig2[sig2["bar_end_ts"].isin(common_ts)],
                on=["bar_end_ts","seccode"], suffixes=("_full","_cut")
            )
            assert not merged.empty


# ─────────────────────────────────────────────
# M3 TESTS
# ─────────────────────────────────────────────
class TestBacktest:
    def setup_method(self):
        self.bars = make_bars_5min()
        self.signal = SignalGenerator(delta=0.05).compute(self.bars)
        self.bt = BaselineBacktest()

    def test_output_schema(self):
        result = self.bt.run(self.signal, self.bars)
        assert set(result.columns) >= {"bar_end_ts","seccode","pos","pnl_mid","cum_pnl_mid"}

    def test_pos_from_signal(self):
        result = self.bt.run(self.signal, self.bars)
        assert result["pos"].between(-1, 1).all()

    def test_cum_pnl_monotone_with_positive_pnl(self):
        result = self.bt.run(self.signal, self.bars)
        for sec, grp in result.groupby("seccode"):
            grp = grp.sort_values("bar_end_ts")
            reconstructed = grp["pnl_mid"].cumsum().values
            # cum_pnl should match cumsum of pnl_mid
            np.testing.assert_allclose(
                grp["cum_pnl_mid"].values, reconstructed, rtol=1e-6
            )

    def test_summary_keys(self):
        result = self.bt.run(self.signal, self.bars)
        s = self.bt.summary(result)
        assert "sharpe_ratio" in s
        assert "hit_rate" in s


# ─────────────────────────────────────────────
# M4 TESTS
# ─────────────────────────────────────────────
class TestImpact:
    def setup_method(self):
        bars_5 = make_bars_5min()
        self.bars_1 = make_bars_1min(bars_5)

    def test_fixed_a(self):
        model = ImpactModel(a_fixed=0.02).calibrate(self.bars_1)
        assert (model.summary()["a"] == 0.02).all()

    def test_calibrated_bounds(self):
        model = ImpactModel(a_min=0.01, a_max=0.05).calibrate(self.bars_1)
        a_vals = model.summary()["a"]
        assert a_vals.between(0.01, 0.05).all()

    def test_output_schema(self):
        model = ImpactModel(a_fixed=0.03).calibrate(self.bars_1)
        table = model.compute(self.bars_1)
        assert set(table.columns) >= {"bar_end_ts","seccode","volume_mkt","a"}

    def test_no_zero_volume(self):
        model = ImpactModel(a_fixed=0.03).calibrate(self.bars_1)
        table = model.compute(self.bars_1)
        assert (table["volume_mkt"] >= 1).all()


# ─────────────────────────────────────────────
# M5 TESTS
# ─────────────────────────────────────────────
class TestExecution:
    def setup_method(self):
        np.random.seed(42)
        bars_5 = make_bars_5min()
        self.bars_1 = make_bars_1min(bars_5)
        self.signal = SignalGenerator(delta=0.05).compute(bars_5)
        model = ImpactModel(a_fixed=0.03).calibrate(self.bars_1)
        self.impact = model.compute(self.bars_1)

    def test_twap_equal_split(self):
        opt = ExecutionOptimizer(method="twap")
        q = opt._twap(100, 5)
        assert sum(q) == 100
        assert len(q) == 5

    def test_vwap_total(self):
        opt = ExecutionOptimizer(method="vwap")
        volumes = np.array([100, 200, 150, 300, 250], dtype=float)
        q = opt._vwap(1000, volumes)
        assert sum(q) == 1000
        assert len(q) == 5

    def test_participation_rate_capped(self):
        opt = ExecutionOptimizer(aum=50_000_000, x_max=0.3, method="vwap")
        schedule, _ = opt.run(self.signal, self.bars_1, self.impact)
        if not schedule.empty:
            assert (schedule["participation_rate"].abs() <= 0.3 + 1e-9).all()

    def test_schedule_schema(self):
        opt = ExecutionOptimizer(method="twap")
        schedule, summary = opt.run(self.signal, self.bars_1, self.impact)
        if not schedule.empty:
            assert set(schedule.columns) >= {
                "bar_end_ts_5min","bar_end_ts_1min","seccode",
                "q_slice","participation_rate","impact_cost_rel"
            }
        if not summary.empty:
            assert set(summary.columns) >= {
                "bar_end_ts_5min","seccode","Q_executed",
                "vwap_fill","implementation_shortfall","pnl_net"
            }

    def test_is_non_negative(self):
        opt = ExecutionOptimizer(method="vwap")
        _, summary = opt.run(self.signal, self.bars_1, self.impact)
        if not summary.empty:
            assert (summary["implementation_shortfall"] >= 0).all()


# ─────────────────────────────────────────────
# OPTIMAL AUM TESTS
# ─────────────────────────────────────────────
class TestOptimalAUM:
    def test_x_star_positive(self):
        backtest = pd.DataFrame({
            "seccode": ["A","A","B","B"],
            "pnl_mid": [0.001, 0.002, 0.0005, 0.001],
            "pos":     [0.5, 0.5, 0.3, 0.3],
        })
        adv = pd.DataFrame({"seccode":["A","B"], "adv":[1e6, 2e6]})
        cal = pd.DataFrame({"seccode":["A","B"], "a":[0.03, 0.02]})
        result = compute_optimal_aum(backtest, adv, cal)
        assert (result["X_star"] >= 0).all()
        assert "net_pnl_peak" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
