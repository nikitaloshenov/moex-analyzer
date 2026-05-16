"""
Optimal AUM Estimation
======================
For each ticker s:

    f(X) = alpha_s * X - (a_s / ADV_s) * X^2

    X*_s = alpha_s * ADV_s / (2 * a_s)

Portfolio X* = Σ_s X*_s (with position-sign constraints from signal)
"""

import numpy as np
import pandas as pd


def compute_optimal_aum(
    backtest_baseline: pd.DataFrame,
    adv:              pd.DataFrame,
    impact_calibrated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    backtest_baseline  : DataFrame[seccode, pnl_mid, pos]
    adv                : DataFrame[seccode, adv]
    impact_calibrated  : DataFrame[seccode, a]

    Returns
    -------
    DataFrame[seccode, alpha, adv, a, X_star, net_pnl_curve_peak]
    """
    # Estimate alpha = mean pnl_mid per bar (when position ≠ 0)
    alpha_df = (
        backtest_baseline[backtest_baseline["pos"] != 0]
        .groupby("seccode")["pnl_mid"]
        .mean()
        .reset_index()
        .rename(columns={"pnl_mid": "alpha"})
    )

    merged = (
        alpha_df
        .merge(adv, on="seccode", how="left")
        .merge(impact_calibrated[["seccode", "a"]], on="seccode", how="left")
    )
    merged["adv"]   = merged["adv"].fillna(1e6)
    merged["a"]     = merged["a"].fillna(0.03)
    merged["alpha"] = merged["alpha"].clip(lower=0)  # only profitable signals

    # X* = alpha * ADV / (2a)
    merged["X_star"] = merged["alpha"] * merged["adv"] / (2 * merged["a"])

    # Peak net PnL at X*: f(X*) = alpha * X* - (a/ADV) * X*^2
    c = merged["a"] / merged["adv"]
    merged["net_pnl_peak"] = merged["alpha"] * merged["X_star"] - c * merged["X_star"]**2

    return merged[["seccode", "alpha", "adv", "a", "X_star", "net_pnl_peak"]].sort_values(
        "X_star", ascending=False
    ).reset_index(drop=True)
