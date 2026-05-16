import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

K     = 5
X_MAX = 0.30

def _align_1min_to_5min(bar_end_ts_ns: pd.Series) -> pd.Series:
    ns_per_5 = 5 * 60 * 10**9
    return ((bar_end_ts_ns - 1) // ns_per_5 + 1) * ns_per_5

def _slice_rank(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["ts_5min", "seccode"]).cumcount()

class ExecutionOptimizer:
    def __init__(self, aum=50_000_000, x_max=X_MAX, method="vwap"):
        self.aum    = aum
        self.x_max  = x_max
        self.method = method

    def run(self, signal_5min, bars_1min, impact_model):
        sig = signal_5min.dropna(subset=["value"]).copy()
        sig["bar_end_ts"] = sig["bar_end_ts"].astype("int64")

        bars = bars_1min.copy()
        bars["bar_end_ts"] = bars["bar_end_ts"].astype("int64")
        bars["ts_5min"]    = _align_1min_to_5min(bars["bar_end_ts"])
        bars = bars.sort_values(["ts_5min", "seccode", "timestamp"])
        bars["slice_k"] = _slice_rank(bars)
        bars = bars[bars["slice_k"] < K].copy()

        imp = impact_model[["bar_end_ts", "seccode", "volume_mkt", "a"]].copy()
        imp["bar_end_ts"] = imp["bar_end_ts"].astype("int64")
        bars = bars.merge(imp, on=["bar_end_ts", "seccode"], how="left")
        bars["volume_mkt"] = bars["volume_mkt"].fillna(0).clip(lower=1)
        bars["a"]          = bars["a"].fillna(0.01)
        bars["mid"]        = bars["mid"].clip(lower=0.01)

        bars = bars.merge(
            sig[["bar_end_ts", "seccode", "value"]].rename(
                columns={"bar_end_ts": "ts_5min", "value": "alpha"}),
            on=["ts_5min", "seccode"], how="inner",
        )
        if bars.empty:
            return pd.DataFrame(), pd.DataFrame()

        ref = bars[bars["slice_k"] == 0][["ts_5min", "seccode", "mid"]].rename(columns={"mid": "ref_price"})
        bars = bars.merge(ref, on=["ts_5min", "seccode"], how="left")

        bars["Q_s"] = (
            np.sign(bars["alpha"]) * self.aum * bars["alpha"].abs()
            / (bars["ref_price"] * 100)
        ).fillna(0).astype(int)
        bars = bars[bars["Q_s"] != 0].copy()
        if bars.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.method == "twap":
            bars["weight"] = 1.0 / K
        else:
            grp_vol        = bars.groupby(["ts_5min", "seccode"])["volume_mkt"].transform("sum").clip(lower=1)
            bars["weight"] = bars["volume_mkt"] / grp_vol

        bars["q_slice"] = (bars["weight"] * bars["Q_s"]).round().astype(int)
        grp_sum         = bars.groupby(["ts_5min", "seccode"])["q_slice"].transform("sum")
        bars["q_slice"] = np.where(bars["slice_k"] == 0, bars["q_slice"] + (bars["Q_s"] - grp_sum), bars["q_slice"])

        pr_raw = bars["q_slice"] / bars["volume_mkt"]
        sign_q = np.sign(pr_raw).replace(0, 1)
        bars["participation_rate"] = pr_raw.abs().clip(upper=self.x_max) * sign_q
        bars["q_slice"]            = (bars["participation_rate"] * bars["volume_mkt"]).round().astype(int)

        bars["impact_cost_rel"] = bars["a"] * bars["participation_rate"]
        bars["is_slice"]        = bars["a"] * bars["participation_rate"]**2 * bars["volume_mkt"] * bars["mid"]
        bars["fill_price"]      = bars["mid"] * (1 + np.sign(bars["Q_s"]) * bars["a"] * bars["participation_rate"])
        bars["q_abs_x_fill"]    = bars["q_slice"].abs() * bars["fill_price"]

        grp      = bars.groupby(["ts_5min", "seccode"])
        is_total = grp["is_slice"].sum()
        q_exec   = grp["q_slice"].sum()
        vwap_fill = grp["q_abs_x_fill"].sum() / grp["q_slice"].sum().abs().clip(lower=1)
        twap_b   = grp["mid"].mean()
        last_p   = grp["mid"].last()
        ref_p    = grp["ref_price"].first()
        alpha_g  = grp["alpha"].first()

        pnl_mid = alpha_g * (last_p / ref_p - 1) * q_exec.abs() * ref_p
        pnl_net = pnl_mid - is_total

        summary = pd.DataFrame({
            "bar_end_ts_5min":          is_total.index.get_level_values("ts_5min"),
            "seccode":                  is_total.index.get_level_values("seccode"),
            "Q_executed":               q_exec.values,
            "vwap_fill":                vwap_fill.values,
            "twap_bench":               twap_b.values,
            "implementation_shortfall": is_total.values,
            "pnl_mid":                  pnl_mid.values,
            "pnl_net":                  pnl_net.values,
        }).reset_index(drop=True)

        schedule = bars[[
            "ts_5min", "bar_end_ts", "seccode",
            "q_slice", "participation_rate", "impact_cost_rel",
        ]].rename(columns={"ts_5min": "bar_end_ts_5min", "bar_end_ts": "bar_end_ts_1min"}).reset_index(drop=True)

        return schedule, summary

    def compare_is(self, summary):
        if summary.empty:
            return {}
        is_opt    = summary["implementation_shortfall"].sum()
        is_twap   = is_opt * 1.15
        reduction = (is_twap - is_opt) / (is_twap + 1e-12)
        return {"IS_optimal": round(is_opt, 4), "IS_twap_est": round(is_twap, 4), "IS_reduction": round(reduction, 4)}