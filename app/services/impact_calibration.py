"""
Калибровка market impact по истории (заготовка).

Идея: по наблюдениям (доля номинала к обороту → реализованное проскальзывание)
подобрать коэффициенты вместо фиксированных 0.1–0.5% из config.
"""

from __future__ import annotations

from typing import Sequence


# -----------------------------------------------------------------------------
# Stub-регрессия (без sklearn — только формула для демо)
# -----------------------------------------------------------------------------


def fit_linear_impact_stub(
    notional_to_turnover_ratio: Sequence[float],
    observed_slip_fraction: Sequence[float],
) -> dict[str, float | str]:
    """
    Минимальные квадраты y ~ a*x + b вручную (для пары точек).

    Для продакшена: sklearn.linear_model.Ridge + кросс-валидация по дням.
    """
    xs = list(notional_to_turnover_ratio)
    ys = list(observed_slip_fraction)
    if len(xs) < 2 or len(xs) != len(ys):
        return {"a": 0.0, "b": 0.0005, "status": "insufficient_data"}

    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    if var_x < 1e-18:
        return {"a": 0.0, "b": float(my), "status": "degenerate_x"}
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    a = cov / var_x
    b = my - a * mx
    return {"a": float(a), "b": float(b), "status": "ok", "note": "demo OLS; ограничьте a,b физически при подстановке"}
