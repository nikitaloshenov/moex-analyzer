"""
Калькулятор доходности для UI (Streamlit).

Связь с бэктестом: оценка market impact по обороту — та же формула, что в бэктесте,
через `get_turnover_slip_fraction` (настройки в app.core.config.settings).
"""

from __future__ import annotations

from app.services.backtest import get_turnover_slip_fraction


# -----------------------------------------------------------------------------
# TradingCalc
# -----------------------------------------------------------------------------


class TradingCalc:
    """Сценарии годовой доходности на капитал (демо). Impact уменьшает показанную прибыль."""

    AVG_YEARLY_RETURN = 0.55
    POTENTIAL_RETURN = 1.20

    @staticmethod
    def calculate_income(investment: float, is_potential: bool = False) -> float:
        rate = TradingCalc.POTENTIAL_RETURN if is_potential else TradingCalc.AVG_YEARLY_RETURN
        return round(float(investment) * rate, 2)

    @staticmethod
    def calculate_investment(target_profit: float, is_potential: bool = False) -> float:
        rate = TradingCalc.POTENTIAL_RETURN if is_potential else TradingCalc.AVG_YEARLY_RETURN
        return round(float(target_profit) / rate, 2)

    @staticmethod
    def slip_fraction_from_turnover(
        position_notional_rub: float,
        avg_daily_turnover_rub: float,
        execution_mode: str = "immediate",
    ) -> float:
        """Доля проскальзывания к цене (0.002 = 0.2%)."""
        if position_notional_rub <= 0 or avg_daily_turnover_rub <= 0:
            return 0.0
        return float(
            get_turnover_slip_fraction(
                position_notional_rub,
                avg_daily_turnover_rub,
                execution_mode,
            )
        )

    @staticmethod
    def demo_dampening_factor(slip_frac: float) -> float:
        """
        Насколько уменьшить показанную прибыль при заданном slip (только для демо-экрана).

        slip_frac — результат slip_fraction_from_turnover; k подобран вручную под наглядность.
        """
        k = 120.0
        return max(0.15, 1.0 - min(0.92, slip_frac * k))

    @staticmethod
    def dampen_display_profit(profit_rub: float, slip_frac: float) -> float:
        """Применить demo_dampening_factor к уже посчитанной сумме прибыли, ₽."""
        return round(float(profit_rub) * TradingCalc.demo_dampening_factor(slip_frac), 2)
