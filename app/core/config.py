"""
Глобальные настройки (pydantic-settings).

Схема параметров для pipeline, impact, исполнения, очистки, риск-менеджмента,
метрик и паттернов. Сервисы `analyzer` и `backtest` читают `settings`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Вложенные блоки настроек (логические группы)
# -----------------------------------------------------------------------------


class IngestionConfig(BaseModel):
    """Сырые события → свечи (Orderlog / тики), агрегация в Polars."""

    resample_interval: str = Field(
        default="1m",
        description="Интервал ресемплинга тиков в OHLCV (напр. 1m, 5m, 1h)",
    )
    polars_streaming: bool = Field(default=True, description="Использовать streaming/lazy Polars на больших логах")


class PatternSignalsConfig(BaseModel):
    """Пороги PatternAnalyzer (вместо магических чисел в коде)."""

    atr_window: int = Field(default=20, ge=5, description="Окно ATR")
    tp_mult: float = Field(default=2.5, ge=0.5, description="Множитель TP от ATR")
    sl_mult: float = Field(default=1.5, ge=0.5, description="Множитель SL от ATR")
    volume_confirm_multiplier: float = Field(
        default=1.2, ge=1.0, description="Текущий объём > среднего × множитель (VSA)"
    )


class DataCleaningConfig(BaseModel):
    """Фильтрация выбросов перед PatternAnalyzer / бэктестом."""

    enabled: bool = Field(default=True)
    ohlc_max_body_to_range_ratio: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Подозрительные доджи/спайки по отношению тела к диапазону",
    )
    volume_spike_zscore: float = Field(
        default=5.0,
        ge=2.0,
        description="Порог по Z-score для объёмного выброса (по окну)",
    )
    volume_spike_window: int = Field(default=20, ge=5, description="Окно для среднего объёма при детекции спайка")


class TurnoverMarketImpactConfig(BaseModel):
    """
    Импакт от крупной позиции относительно среднего дневного оборота (руб).

    Если notional_order / avg_daily_turnover > threshold_pct — ухудшение входа на slip_pct.
    (Отдельно от dataclass MarketImpactConfig в backtest.py — на этапе 2 сведём в одну модель.)
    """

    turnover_threshold_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=100.0,
        description="Порог доли заявки от среднего дневного оборота, %",
    )
    slip_min_pct: float = Field(default=0.1, ge=0.0, description="Мин. ухудшение цены входа, %")
    slip_max_pct: float = Field(default=0.5, ge=0.0, description="Макс. ухудшение цены входа, %")
    avg_turnover_lookback_days: int = Field(default=20, ge=5, description="Дней для оценки среднего дневного оборота")

    @model_validator(mode="after")
    def slip_order(self) -> TurnoverMarketImpactConfig:
        if self.slip_max_pct < self.slip_min_pct:
            raise ValueError("slip_max_pct должен быть >= slip_min_pct")
        return self


class ExecutionConfig(BaseModel):
    """Режим исполнения крупной заявки (кейс: оптимальное распределение во времени)."""

    default_mode: Literal["immediate", "twap", "vwap"] = Field(
        default="immediate",
        description="Сразу | TWAP (равные доли по времени) | VWAP (вес по объёму)",
    )
    twap_slices: int = Field(default=10, ge=2, le=500, description="Число частей TWAP")
    twap_interval_seconds: int = Field(default=60, ge=1, description="Интервал между частями TWAP, сек")
    vwap_volume_lookback_bars: int = Field(default=20, ge=5, description="Окно объёма для весов VWAP")


class RiskManagementConfig(BaseModel):
    """Риск на сделку и лимиты (не пункты, а деньги / доля капитала)."""

    risk_per_trade_pct: float = Field(
        default=1.0,
        ge=0.01,
        le=100.0,
        description="Макс. риск на сделку в % от капитала (под стоп)",
    )
    max_position_rub: float | None = Field(
        default=None,
        ge=0,
        description="Жёсткий потолок номинала позиции, руб (None = без лимита)",
    )
    max_daily_loss_pct: float | None = Field(
        default=None,
        ge=0.1,
        le=100.0,
        description="Дневной стоп по просадке портфеля, % (None = выкл)",
    )
    max_open_trades: int = Field(default=5, ge=1, le=100, description="Лимит одновременно открытых сделок в бэктесте")


class BacktestMetricsConfig(BaseModel):
    """Метрики после серии сделок (Sharpe, Sortino, просадка)."""

    risk_free_rate_annual: float = Field(default=0.0, description="Безрисковая ставка годовых, доля единицы")
    periods_per_year: int = Field(default=252, ge=1, description="Торговых периодов в году для годовизации")


class RegimeConfig(BaseModel):
    """Фильтр режима рынка (флэт / тренд) до входа в сделку."""

    enabled: bool = Field(default=True, description="Включить фильтр по режиму")
    ma_fast: int = Field(default=10, ge=3, description="Быстрая MA по close")
    ma_slow: int = Field(default=30, ge=5, description="Медленная MA по close")
    flat_band: float = Field(
        default=0.008,
        ge=0.001,
        le=0.05,
        description="Если |MAf-MAs|/close < band — считаем рынок боковиком (flat)",
    )
    allow_strong_in_flat: bool = Field(
        default=True,
        description="Во флэте разрешать только STRONG BUY/STRONG SELL (если True)",
    )


class PortfolioConfig(BaseModel):
    """Мульти-тикер: агрегация и будущие лимиты по корреляции."""

    enabled: bool = Field(default=False, description="Включить расширенные проверки портфеля (пока заготовка)")
    max_pairwise_abs_corr: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Порог |корреляции| между рядами доходностей (stub, пока не блокирует)",
    )


class LimitsConfig(BaseModel):
    """Лимиты API / нагрузки (заготовка под MOEX и кэш)."""

    moex_max_concurrent_requests: int = Field(default=10, ge=1, le=50)
    request_timeout_seconds: float = Field(default=30.0, ge=5.0)
    enable_response_cache: bool = Field(default=True, description="Кэш ответов до появления PostgreSQL")


class CacheConfig(BaseModel):
    """Кэш и БД (этап позже: SQLite → PostgreSQL)."""

    database_url: str = Field(
        default="sqlite+aiosqlite:///moex_database.db",
        description="DSN для кэша котировок (позже можно заменить на PostgreSQL)",
    )
    postgres_url: str | None = Field(default=None, description="Запасной DSN PostgreSQL, если включите миграцию")


# -----------------------------------------------------------------------------
# Корневые настройки приложения
# -----------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Чтение из переменных окружения с префиксом (по умолчанию нет — имена полей как ENV).

    Вложенные модели: через двойное подчёркивание, например:
    MARKET_IMPACT__SLIP_MAX_PCT=0.4
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "MOEX Analyzer"

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
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @field_validator(
        "pattern_signals",
        "ingestion",
        "data_cleaning",
        "market_impact",
        "execution",
        "risk",
        "backtest_metrics",
        "regime",
        "portfolio",
        "limits",
        "cache",
        mode="before",
    )
    @classmethod
    def _coerce_nested(cls, v, info):
        """Разрешить dict из окружения / ручной передачи как вложенный объект."""
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
            "cache": CacheConfig,
        }.get(info.field_name)
        return model(**v) if model else v


settings = Settings()
