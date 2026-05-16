# Trading Signal Pipeline — Хакатон

Полная реализация pipeline от сигнала до оценки исполнения с рыночным импактом.

---

## Структура проекта

```
hackathon/
├── main.py                    # Точка входа — запускает весь pipeline
├── modules/
│   ├── data_loader.py         # M1: Загрузка и квантизация parquet-данных
│   ├── m2_signal.py           # M2: Генерация сигнала [-1, 1]
│   ├── m3_backtest.py         # M3: Baseline бэктест (нулевой импакт)
│   ├── m4_impact.py           # M4: Модель рыночного импакта (Kyle λ)
│   ├── m5_execution.py        # M5: Оптимизация исполнения (TWAP/VWAP/Numeric)
│   └── optimal_aum.py         # Оценка оптимального AUM (X*)
├── tests/
│   └── test_modules.py        # 18 unit-тестов (все проходят)
└── output/                    # Создаётся автоматически
```

---

## Быстрый старт

### 1. Установить зависимости
```bash
pip install pandas numpy scipy pyarrow pytz pytest
```

### 2. Запустить pipeline
```bash
python main.py \
    --folder_1min ./is_features_1_min_hackaton \
    --folder_5min ./if_features_5_min_hackaton \
    --aum 50000000 \
    --method vwap \
    --output ./output
```

### 3. Запустить тесты
```bash
python -m pytest tests/test_modules.py -v
```

---

## Формат входных данных

Ожидаемая структура папок:

```
if_features_5_min_hackaton/
    open.parquet     ← index: timestamp, columns: tickers (ABIO, SBER, ...)
    high.parquet
    low.parquet
    close.parquet
    volume.parquet
    [bid.parquet]    ← опционально (улучшает сигнал)
    [ask.parquet]    ← опционально

is_features_1_min_hackaton/
    open.parquet
    high.parquet
    low.parquet
    close.parquet
    volume.parquet
    [spread.parquet] ← опционально
```

Каждый файл — wide-format:
```
timestamp                      ABIO    SBER    GAZP  ...
2024-09-02 10:01:00+03:00      83.50  267.8   168.3
2024-09-02 10:02:00+03:00      83.55  268.1   168.1
...
```

---

## Описание модулей

### M1 — DataLoader
- Загружает все `.parquet` из обеих папок
- Конвертирует wide → long формат: `[timestamp, seccode, value]`
- Фильтрует по сессии 10:00–18:30 МСК
- Вычисляет `mid = (open + close) / 2`
- Считает ADV (средний дневной объём)

### M2 — SignalGenerator
```
alpha = EMA_fast(log_ret) - EMA_slow(log_ret)    # momentum
      + ob_imbalance                              # если есть bid/ask
→ cross-sectional z-score → clip(-1, 1) → threshold δ
```
- Параметры: `ema_fast=5`, `ema_slow=20`, `delta=0.1`
- Выход: `signal_5min` с полями `[bar_end_ts, seccode, value]`

### M3 — BaselineBacktest
- `pos_t = alpha_{t-1}` (сигнал генерируется в конце бара → вход в следующем)
- `pnl_mid_t = pos_t * (close_t / open_t - 1)`
- Метрики: Sharpe, Hit Rate, cum PnL

### M4 — ImpactModel
Два режима:
- **Fixed**: `a = const` (задаётся через `--a_fixed`)
- **Kyle λ calibration**: OLS регрессия `|Δprice| ~ participation_rate` по историческим барам
- Диапазон: `a ∈ [0.01, 0.05]`

### M5 — ExecutionOptimizer
Три стратегии исполнения:

| Метод | Описание | Когда использовать |
|-------|----------|-------------------|
| `twap` | Равномерное разбиение Q/K | Baseline |
| `vwap` | Пропорционально объёму | По умолчанию (почти оптимально) |
| `numeric` | SciPy SLSQP (полная оптимизация) | Когда нужна точность |

Метрики:
- **IS** = Σ_k a_k × x_{s,k}² × V_{s,k} × P_{s,k}
- **pnl_net** = pnl_mid − IS
- **VWAP fill** vs **TWAP benchmark**

### Optimal AUM
```
X* = α * ADV / (2a)
```
где α — ожидаемая доходность из бэктеста, a — коэф. импакта.

---

## Выходные файлы (./output/)

| Файл | Содержимое |
|------|-----------|
| `signal_5min.parquet` | Сигналы по всем барам |
| `backtest_baseline.parquet` | PnL по mid без импакта |
| `impact_model.parquet` | λ(t) по каждому 1-мин бару |
| `impact_calibration.parquet` | Калиброванный `a` по тикерам |
| `execution_schedule.parquet` | Расписание по слайсам |
| `execution_summary.parquet` | IS, pnl_net, VWAP fill по барам |
| `optimal_aum.parquet` | X* по каждому тикеру |
| `pipeline_report.json` | Сводные метрики всего pipeline |

---

## Параметры CLI

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--folder_1min` | — | Путь к 1-мин данным (обязательно) |
| `--folder_5min` | — | Путь к 5-мин данным (обязательно) |
| `--aum` | 50_000_000 | AUM в рублях |
| `--method` | vwap | Метод исполнения: twap/vwap/numeric |
| `--delta` | 0.1 | Порог сигнала |
| `--ema_fast` | 5 | EMA быстрый период |
| `--ema_slow` | 20 | EMA медленный период |
| `--a_fixed` | None | Фиксированный коэф. импакта (пропустить калибровку) |
| `--output` | ./output | Папка для выходных файлов |
