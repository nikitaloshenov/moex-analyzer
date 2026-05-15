from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# Вложенные блоки настроек
# -----------------------------------------------------------------------------

class IngestionConfig(BaseModel):
    """Сырые события → свечи (Orderlog / тики), агрегация в Polars."""
    resample_interval: str = Field(default="1m", description="Интервал ресемплинга тиков")
    polars_streaming: bool = Field(default=True, description="Использовать streaming/lazy Polars")


class PatternSignalsConfig(BaseModel):
    """Пороги PatternAnalyzer (вместо магических чисел)."""
    atr_window: int = Field(default=20, ge=5, description="Окно ATR")
    tp_mult: float = Field(default=2.5, ge=0.5, description="Множитель TP от ATR")
    sl_mult: float = Field(default=1.5, ge=0.5, description="Множитель SL от ATR")
    volume_confirm_multiplier: float = Field(default=1.2, ge=1.0, description="Множитель VSA")


class DataCleaningConfig(BaseModel):
    """Фильтрация выбросов перед бэктестом."""
    enabled: bool = Field(default=True)
    ohlc_max_body_to_range_ratio: float = Field(default=0.95, ge=0.5, le=1.0)
    volume_spike_zscore: float = Field(default=5.0, ge=2.0)
    volume_spike_window: int = Field(default=20, ge=5)


class TurnoverMarketImpactConfig(BaseModel):
    """
    Модуль М4: Нелинейный Market Impact.
    Формула: IS = coefficient_a * (x^2) * P * q
    """
    coefficient_a: float = Field(
        default=0.03, 
        ge=0.0, 
        description="Коэффициент 'а' квадратичного штрафа со слайдов организаторов"
    )
    # Оставляем старые поля как опциональные, чтобы не упал фронтенд парней
    turnover_threshold_pct: float = Field(default=5.0, ge=0.1, le=100.0)
    slip_min_pct: float = Field(default=0.1, ge=0.0)
    slip_max_pct: float = Field(default=0.5, ge=0.0)
    avg_turnover_lookback_days: int = Field(default=20, ge=5)

    @model_validator(mode="after")
    def slip_order(self) -> TurnoverMarketImpactConfig:
        if self.slip_max_pct < self.slip_min_pct:
            raise ValueError("slip_max_pct должен быть >= slip_min_pct")
        return self


class ExecutionConfig(BaseModel):
    """
    Модуль М5: Настройки TWAP исполнения.
    Параметры дробления и объема крупной заявки.
    """
    default_mode: Literal["immediate", "twap", "vwap"] = Field(
        default="twap",  # Ставим дефолтом twap, под ТЗ
        description="Immediate | TWAP | VWAP",
    )
    order_size_lots: float = Field(
        default=10000.0, 
        ge=1.0, 
        description="Размер крупного ордера Q (в лотах) для симуляции TWAP"
    )
    twap_slices: int = Field(default=30, ge=2, le=500, description="Число частей TWAP (под 30-минутные бары)")
    twap_interval_seconds: int = Field(default=60, ge=1, description="Интервал между частями TWAP (1 минута)")
    vwap_volume_lookback_bars: int = Field(default=20, ge=5)


class RiskManagementConfig(BaseModel):
    """Риск на сделку и лимиты."""
    risk_per_trade_pct: float = Field(default=1.0, ge=0.01, le=100.0)
    max_position_rub: float | None = Field(default=None, ge=0)
    max_daily_loss_pct: float | None = Field(default=None, ge=0.1, le=100.0)
    max_open_trades: int = Field(default=5, ge=1, le=100)


class BacktestMetricsConfig(BaseModel):
    """Метрики после серии сделок."""
    risk_free_rate_annual: float = Field(default=0.0)
    periods_per_year: int = Field(default=252, ge=1)


class RegimeConfig(BaseModel):
    """Фильтр режима рынка (флэт / тренд)."""
    enabled: bool = Field(default=True)
    ma_fast: int = Field(default=10, ge=3)
    ma_slow: int = Field(default=30, ge=5)
    flat_band: float = Field(default=0.008, ge=0.001, le=0.05)
    allow_strong_in_flat: bool = Field(default=True)


class PortfolioConfig(BaseModel):
    enabled: bool = Field(default=False)
    max_pairwise_abs_corr: float = Field(default=0.85, ge=0.0, le=1.0)


class LimitsConfig(BaseModel):
    moex_max_concurrent_requests: int = Field(default=10, ge=1, le=50)
    request_timeout_seconds: float = Field(default=30.0, ge=5.0)    


# -----------------------------------------------------------------------------
# Корневой класс настроек Settings
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "MOEX Analyzer"

    pattern_signals: PatternSignalsConfig = Field(default_factory=PatternSignalsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    data_cleaning: DataCleaningConfig = Field(default_factory=DataCleaningConfig)
    market_impact: TurnoverMarketImpactConfig = Field(default_factory=TurnoverMarketImpactConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    backtest_metrics: BacktestMetricsConfig = Field(default_factory=BacktestMetricsConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @field_validator(
        "pattern_signals", "ingestion", "data_cleaning", "market_impact",
        "execution", "risk", "backtest_metrics", "regime", "portfolio", "limits",  
        mode="before",
    )
    @classmethod
    def _coerce_nested(cls, v, info):
        if v is None or not isinstance(v, dict):
            return v
        model = {
            "pattern_signals": PatternSignalsConfig,
            "ingestion": IngestionConfig,
            "data_cleaning": DataCleaningConfig,
            "market_impact": TurnoverMarketImpactConfig,
            "execution": ExecutionConfig,
            "risk": RiskManagementConfig,
            "backtest_metrics": BacktestMetricsConfig,
            "regime": RegimeConfig,
            "portfolio": PortfolioConfig,
            "limits": LimitsConfig,            
        }.get(info.field_name)
        return model(**v) if model else v


settings = Settings()