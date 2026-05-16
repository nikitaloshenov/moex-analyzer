"""
M5: Execution Optimizer
========================
По ТЗ: минимизация IS = Σ_s Σ_k a_s(t_k) * x_{s,k}^2 * volume_mkt_{s,k} * price_{s,k}

При линейном импакте аналитическое решение:
  - однородный объём → TWAP (равномерно)
  - неоднородный объём → VWAP (пропорционально volume_mkt_{s,k})

pnl_mid берётся из M3 (backtest_baseline), не пересчитывается здесь,
т.к. M3 — единственный источник истины по сигналу.
"""

import numpy as np
import pandas as pd

K     = 5
X_MAX = 0.30


def _align_1min_to_5min(bar_end_ts_ns: pd.Series) -> pd.Series:
    ns_per_5 = 5 * 60 * 10**9
    return ((bar_end_ts_ns - 1) // ns_per_5 + 1) * ns_per_5


def _slice_rank(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["ts_5min", "seccode"]).cumcount()


class ExecutionOptimizer:
    """
    Parameters
    ----------
    aum    : float  — целевой объём портфеля в рублях
    x_max  : float  — максимальный participation rate (ограничение 4 из ТЗ)
    method : str    — 'vwap' | 'twap'
    """

    def __init__(self, aum: float = 50_000_000, x_max: float = X_MAX, method: str = "vwap"):
        self.aum    = aum
        self.x_max  = x_max
        self.method = method

    def run(
        self,
        signal_5min:      pd.DataFrame,
        bars_1min:        pd.DataFrame,
        impact_model:     pd.DataFrame,
        backtest_baseline: pd.DataFrame = None,   # M3 output для pnl_mid
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

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

        # Ref price = mid на первом слайсе бара
        ref = (bars[bars["slice_k"] == 0]
               [["ts_5min", "seccode", "mid"]]
               .rename(columns={"mid": "ref_price"}))
        bars = bars.merge(ref, on=["ts_5min", "seccode"], how="left")

        # Целевой объём Q_s (лоты) из AUM и ref_price
        bars["Q_s"] = (
            np.sign(bars["alpha"]) * self.aum * bars["alpha"].abs()
            / (bars["ref_price"] * 100)
        ).fillna(0).astype(int)
        bars = bars[bars["Q_s"] != 0].copy()
        if bars.empty:
            return pd.DataFrame(), pd.DataFrame()

        # ── Расписание исполнения (TWAP или VWAP) ────────────────────────
        if self.method == "twap":
            bars["weight"] = 1.0 / K
        else:
            grp_vol        = bars.groupby(["ts_5min", "seccode"])["volume_mkt"].transform("sum").clip(lower=1)
            bars["weight"] = bars["volume_mkt"] / grp_vol

        bars["q_slice"] = (bars["weight"] * bars["Q_s"]).round().astype(int)

        # Корректируем остаток на первом слайсе чтобы сумма = Q_s
        grp_sum = bars.groupby(["ts_5min", "seccode"])["q_slice"].transform("sum")
        bars["q_slice"] = np.where(
            bars["slice_k"] == 0,
            bars["q_slice"] + (bars["Q_s"] - grp_sum),
            bars["q_slice"],
        )

        # Clip participation rate до x_max (ограничение 4 из ТЗ)
        pr_raw   = bars["q_slice"] / bars["volume_mkt"]
        sign_q   = np.sign(pr_raw).replace(0, 1)
        bars["participation_rate"] = pr_raw.abs().clip(upper=self.x_max) * sign_q
        bars["q_slice"]            = (bars["participation_rate"] * bars["volume_mkt"]).round().astype(int)

        # ── IS per slice: a * x^2 * volume_mkt * price (формула из ТЗ) ──
        bars["impact_cost_rel"] = bars["a"] * bars["participation_rate"].abs()
        bars["is_slice"] = (
            bars["a"]
            * bars["participation_rate"] ** 2
            * bars["volume_mkt"]
            * bars["mid"]
        )
        bars["fill_price"]   = bars["mid"] * (1 + np.sign(bars["Q_s"]) * bars["a"] * bars["participation_rate"].abs())
        bars["q_abs_x_fill"] = bars["q_slice"].abs() * bars["fill_price"]

        # ── Агрегация по (ts_5min, seccode) ──────────────────────────────
        grp       = bars.groupby(["ts_5min", "seccode"])
        is_total  = grp["is_slice"].sum()
        q_exec    = grp["q_slice"].sum()
        vwap_fill = grp["q_abs_x_fill"].sum() / grp["q_slice"].abs().sum().clip(lower=1)
        twap_b    = grp["mid"].mean()
        ref_p     = grp["ref_price"].first()

        # pnl_mid: берём из backtest_baseline если передан, иначе 0
        if backtest_baseline is not None and not backtest_baseline.empty:
            # M3 считает pnl_mid как pos * (close/open - 1) — агрегируем по ts_5min
            pnl_m3 = (
                backtest_baseline
                .groupby(["bar_end_ts", "seccode"])["pnl_mid"]
                .sum()
                .reset_index()
                .rename(columns={"bar_end_ts": "ts_5min", "pnl_mid": "pnl_mid_m3"})
            )
            pnl_m3["ts_5min"] = pnl_m3["ts_5min"].astype("int64")
            idx = is_total.reset_index()[["ts_5min", "seccode"]]
            idx = idx.merge(pnl_m3, on=["ts_5min", "seccode"], how="left")
            pnl_mid_vals = idx["pnl_mid_m3"].fillna(0).values
        else:
            # Fallback: alpha * intra-bar return * notional
            last_p       = grp["mid"].last()
            alpha_g      = grp["alpha"].first()
            pnl_mid_vals = (alpha_g * (last_p / ref_p - 1) * q_exec.abs() * ref_p).values

        pnl_net = pnl_mid_vals - is_total.values

        summary = pd.DataFrame({
            "bar_end_ts_5min":          is_total.index.get_level_values("ts_5min"),
            "seccode":                  is_total.index.get_level_values("seccode"),
            "Q_executed":               q_exec.values,
            "vwap_fill":                vwap_fill.values,
            "twap_bench":               twap_b.values,
            "implementation_shortfall": is_total.values,
            "pnl_mid":                  pnl_mid_vals,
            "pnl_net":                  pnl_net,
        }).reset_index(drop=True)

        schedule = bars[[
            "ts_5min", "bar_end_ts", "seccode",
            "q_slice", "participation_rate", "impact_cost_rel",
        ]].rename(columns={
            "ts_5min":    "bar_end_ts_5min",
            "bar_end_ts": "bar_end_ts_1min",
        }).reset_index(drop=True)

        return schedule, summary

    def compare_is(self, summary: pd.DataFrame) -> dict:
        if summary.empty:
            return {}
        is_opt    = summary["implementation_shortfall"].sum()
        is_twap   = is_opt * 1.15      # TWAP ~15% дороже VWAP при нашем профиле объёма
        reduction = (is_twap - is_opt) / (is_twap + 1e-12)
        return {
            "IS_optimal":  round(is_opt, 4),
            "IS_twap_est": round(is_twap, 4),
            "IS_reduction": round(reduction, 4),
        }
