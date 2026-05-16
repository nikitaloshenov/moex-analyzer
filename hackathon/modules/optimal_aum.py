"""
Optimal AUM Estimation
======================
По ТЗ: f(X) = alpha_s * X - (a_s / ADV_s) * X^2

X*_s = alpha_s * ADV_s / (2 * a_s)

ADV считается в рублях (лоты × mid-цена), чтобы X* был в рублях.
alpha — средний pnl_mid per bar при pos ≠ 0, из M3.
"""

import numpy as np
import pandas as pd


def compute_optimal_aum(
    backtest_baseline:  pd.DataFrame,   # [seccode, pnl_mid, pos]
    adv:                pd.DataFrame,   # [seccode, adv]  — в лотах
    impact_calibrated:  pd.DataFrame,   # [seccode, a]
    bars_1min:          pd.DataFrame = None,  # для перевода ADV в рубли
) -> pd.DataFrame:
    """
    Returns
    -------
    DataFrame[seccode, alpha, adv_rub, a, X_star, net_pnl_peak]
    """
    # alpha = средний pnl_mid per bar при ненулевой позиции
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
    merged["adv"] = merged["adv"].fillna(0)
    merged["a"]   = merged["a"].fillna(0.01)

    # ADV в рублях: если bars_1min передан — считаем mid-цену per ticker
    if bars_1min is not None:
        mid_price = (
            bars_1min.groupby("seccode")["mid"]
            .median()
            .reset_index()
            .rename(columns={"mid": "price"})
        )
        merged = merged.merge(mid_price, on="seccode", how="left")
        merged["price"]   = merged["price"].fillna(1.0).clip(lower=0.01)
        merged["adv_rub"] = merged["adv"] * merged["price"]
    else:
        merged["adv_rub"] = merged["adv"]

    merged["adv_rub"] = merged["adv_rub"].clip(lower=1.0)

    # Только тикеры с положительным alpha (прибыльный сигнал)
    merged["alpha"] = merged["alpha"].clip(lower=0)

    # X* = alpha * ADV_rub / (2 * a)  [в рублях]
    merged["X_star"] = merged["alpha"] * merged["adv_rub"] / (2 * merged["a"])

    # f(X*) = alpha * X* - (a / ADV_rub) * X*^2
    c = merged["a"] / merged["adv_rub"]
    merged["net_pnl_peak"] = (
        merged["alpha"] * merged["X_star"]
        - c * merged["X_star"] ** 2
    )

    return (
        merged[["seccode", "alpha", "adv_rub", "a", "X_star", "net_pnl_peak"]]
        .rename(columns={"adv_rub": "adv"})
        .sort_values("X_star", ascending=False)
        .reset_index(drop=True)
    )
