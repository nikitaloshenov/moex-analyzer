import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Корень проекта в PYTHONPATH (импорты app.* и web.*)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from web.calculator import TradingCalc

# --- 1. КОНФИГУРАЦИЯ ---
st.set_page_config(
    page_title="JiDi Alpha • V8.5",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Функция форматирования чисел: 1.000.000
def fmt(value):
    return f"{int(value):,}".replace(",", ".")

# --- 2. CSS (СТРОГИЙ ТЕРМИНАЛ) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

    .stApp {
        background-color: #050505; 
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF;
    }

    /* БЛОК 1 (ЗАМОРОЖЕН) */
    .frozen-h1 { font-size: 3.5rem !important; font-weight: 700; color: #FFFFFF !important; margin-bottom: 5px; letter-spacing: -1.5px; }
    .frozen-sub { color: #AAAAAA; font-weight: 400; font-size: 1.4rem; margin-bottom: 40px; }

    /* БЛОК 2 (КАЛЬКУЛЯТОР) */
    .calc-header { 
        font-size: 3rem !important; 
        font-weight: 700; 
        color: #FFFFFF; 
        margin: 60px 0 35px 0 !important;
        letter-spacing: -1px;
    }

    /* Рамки блоков: 2px толщина, Тёмно-серый (#222) */
    [data-testid="stMetric"], .custom-card {
        background: #0A0A0A !important;
        border: 2px solid #222222 !important; 
        border-radius: 15px !important;
        padding: 30px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    .custom-card:hover { border: 2px solid #282828 !important; }

    /* Метрики из первого блока */
    [data-testid="stMetricValue"] { color: #FFD700 !important; font-size: 2.8rem !important; font-weight: 700 !important;}
    [data-testid="stMetricDelta"] { color: #00FF7F !important; font-weight: 600 !important;}
    [data-testid="stMetricDelta"] svg { fill: #00FF7F !important; }

    /* Слайдеры: Золотая нить (если где-то остались) */
    .stSlider > div > div > div > div { background-color: #FFD700 !important; }
    
    #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. БЛОК 1 (БЕЗ ИЗМЕНЕНИЙ) ---
st.markdown('<h1 class="frozen-h1">Аналитическая платформа MOEX</h1>', unsafe_allow_html=True)
st.markdown('<div class="frozen-sub">Институциональный анализ ликвидности и паттернов.</div>', unsafe_allow_html=True)

def get_chart():
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    equity = 1000 * (1 + np.random.normal(0.0015, 0.007, len(dates))).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', line=dict(color='#FFD700', width=3)))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0), height=350,
        xaxis=dict(showgrid=False, color="#444"),
        yaxis=dict(showgrid=True, gridcolor="#111", color="#444")
    )
    return fig

st.plotly_chart(get_chart(), use_container_width=True)

m1, m2, m3 = st.columns(3)
with m1: st.metric("Чистая прибыль", f"{fmt(3388)}", "пт. / год")
with m2: st.metric("Profit Factor", "3.29", "SBER/GAZP")
with m3: st.metric("Winrate", "40.9%", "Verified")

# --- 4. БЛОК 2 (ЧИСТАЯ НАСТРОЙКА) ---

st.markdown('<h1 style="color: white; margin-top: 60px; margin-bottom: 30px;">Сравнительный анализ прибыли</h1>', unsafe_allow_html=True)

# --- 4. БЛОК 2: ВЫСОКИЙ КОМПАКТНЫЙ КАЛЬКУЛЯТОР ---

# Центрированный заголовок
st.markdown('<h1 style="text-align: center; color: white; margin-top: 60px; margin-bottom: 25px; font-size: 3.2rem;">Сравнительный анализ прибыли</h1>', unsafe_allow_html=True)

# Стили: Узкий (450px), но высокий и массивный
st.markdown("""
    <style>
    /* Центрируем и делаем блок ВЫШЕ */
    div[data-testid="stNumberInput"] {
        background: #111111 !important;
        border: 1px solid #262730 !important;
        border-radius: 15px !important;
        padding: 40px 30px !important; /* Увеличили внутренние отступы для высоты */
        margin: 0 auto !important; 
        width: 450px !important;    /* Вернули компактную ширину */
        min-height: 180px !important; /* Добавили минимальную высоту */
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    div[data-testid="stNumberInput"] > div[data-baseweb="input"] {
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stNumberInput"]:focus-within {
        border-color: #444444 !important;
        box-shadow: 0 0 0 1px #444444 !important;
    }

    div[data-testid="stNumberInput"] label {
        color: #FFD700 !important;
        font-size: 1.1rem !important; 
        font-weight: 700 !important;
        text-transform: uppercase;
        text-align: center !important;
        display: block !important;
        margin-bottom: 20px !important;
        letter-spacing: 1.5px;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 3.2rem !important; /* Еще мощнее шрифт */
        font-weight: 800 !important;
        color: #FFFFFF !important;
        text-align: center !important;
    }

    div[data-testid="stNumberInput"] button { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Обертка для позиционирования (используем широкие края для центрации узкого блока)
_, col_center, _ = st.columns([1, 1, 1])
with col_center:
    user_cap = st.number_input(
        "Укажите объем рабочего капитала:", 
        min_value=1000, 
        max_value=9999999, 
        value=140000, 
        step=10000, 
        key="cap_centered_tall_v13" 
    )

# Ликвидность: те же пороги, что в бэктесте (settings.market_impact)
liq1, liq2, liq3 = st.columns(3)
with liq1:
    pos_rub = st.number_input(
        "Номинал позиции, ₽",
        min_value=0,
        value=0,
        step=100_000,
        key="pos_rub_ui",
    )
with liq2:
    avg_turn = st.number_input(
        "Ср. дневной оборот (оценка), ₽",
        min_value=1.0,
        value=50_000_000.0,
        step=1_000_000.0,
        key="avg_turn_ui",
    )
with liq3:
    ex_mode = st.selectbox(
        "Исполнение заявки",
        ("immediate", "twap", "vwap"),
        index=0,
        key="exec_mode_ui",
    )

sf = TradingCalc.slip_fraction_from_turnover(pos_rub, avg_turn, ex_mode)
damp = TradingCalc.demo_dampening_factor(sf)
st.caption(
    f"Market impact (доля к цене): **{sf * 100:.4f} %**  ·  демо-корректор прибыли на экране: **×{damp:.3f}** "
    "(пороги из `.env` / `config`, формула как в `run_pattern_backtest`)."
)

# Расчеты: 60% / 120% — витрина; отдельно строка «после impact» через калькулятор
p_base = user_cap * 0.60
p_alpha = user_cap * 1.20
p_base_dj = TradingCalc.dampen_display_profit(p_base, sf)
p_alpha_dj = TradingCalc.dampen_display_profit(p_alpha, sf)

st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)

# --- КАРТОЧКИ (БЕЗ ИЗМЕНЕНИЙ ШРИФТА) ---
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown(f"""
        <div style="background: #0A0A0A; padding: 35px; border-radius: 15px; border: 2px solid #222222; min-height: 420px; display: flex; flex-direction: column; border-left: 2px solid #FFD700;">
            <p style="color: #FFD700; font-weight: 700; text-transform: uppercase; margin: 0; font-size: 1rem; letter-spacing: 1px;">Текущая база (60%)</p>
            <h2 style="font-size: 3.5rem; margin: 25px 0; color: white;">+{fmt(int(p_base))} ₽</h2>
            <p style="color: #888; margin: 0; font-size: 1.05rem;">После impact (демо): <span style="color: #CCC;">+{fmt(int(p_base_dj))} ₽</span></p>
            <p style="color: #666; margin: 0; font-size: 1.3rem;">Итого: <span style="color: #AAA;">{fmt(user_cap + int(p_base))} ₽</span></p>
            <div style="margin-top: auto; padding-top: 30px; border-top: 1px solid #222;">
                <p style="color: #FFD700; font-weight: 700; text-transform: uppercase; font-size: 0.85rem; margin-bottom: 12px;">Базовая архитектура:</p>
                <ul style="list-style-type: none; padding: 0; color: #888; font-size: 0.95rem; line-height: 1.8;">
                    <li>• Трендовое следование (MA20/50)</li>
                    <li>• Фильтрация по ATR (волатильность)</li>
                    <li>• Стандартный риск-менеджмент 1:2</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="background: #0A0A0A; padding: 35px; border-radius: 15px; border: 2px solid #222222; min-height: 420px; display: flex; flex-direction: column; border-left: 2px solid #00FF7F;">
            <p style="color: #00FF7F; font-weight: 700; text-transform: uppercase; margin: 0; font-size: 1rem; letter-spacing: 1px;">Максимизация профит-фактора (120%)</p>
            <h2 style="font-size: 3.5rem; margin: 25px 0; color: #00FF7F;">+{fmt(int(p_alpha))} ₽</h2>
            <p style="color: #888; margin: 0; font-size: 1.05rem;">После impact (демо): <span style="color: #99FFCC;">+{fmt(int(p_alpha_dj))} ₽</span></p>
            <p style="color: #666; margin: 0; font-size: 1.3rem;">Итого: <span style="color: #AAA;">{fmt(user_cap + int(p_alpha))} ₽</span></p>
            <div style="margin-top: auto; padding-top: 30px; border-top: 1px solid #222;">
                <p style="color: #00FF7F; font-weight: 700; text-transform: uppercase; font-size: 0.85rem; margin-bottom: 12px;">Механика вектора успеха:</p>
                <ul style="list-style-type: none; padding: 0; color: #888; font-size: 0.95rem; line-height: 1.8;">
                    <li>• Динамический вектор входа </li>
                    <li>• Ребалансировка плеча по индексу ADX</li>
                    <li>• Сжатие стопа при достижении целей</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 5. БЛОК 3: ВЕКТОРЫ СТРАТЕГИЧЕСКОГО РАЗВИТИЯ (ОБНОВЛЕННЫЕ) ---

st.markdown('<h1 style="text-align: center; color: white; margin-top: 80px; margin-bottom: 50px; font-size: 3.2rem;">Векторы стратегического развития</h1>', unsafe_allow_html=True)

col_v1, col_v2, col_v3 = st.columns(3, gap="large")

with col_v1:
    st.markdown("""
        <div style="background: #0A0A0A; padding: 35px; border-radius: 15px; border: 2px solid #222; min-height: 400px; display: flex; flex-direction: column;">
            <div style="color: #FFD700; font-size: 2rem; margin-bottom: 20px;">◈</div>
            <h3 style="color: white; font-size: 1.6rem; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">Интеграция торгового API</h3>
            <p style="color: #888; font-size: 1.15rem; line-height: 1.7;">
                Развертывание шлюзов для прямого подключения к брокерским терминалам. Переход к модели автоследования для мгновенного исполнения сигналов без проскальзываний.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_v2:
    st.markdown("""
        <div style="background: #0A0A0A; padding: 35px; border-radius: 15px; border: 2px solid #222; min-height: 400px; display: flex; flex-direction: column;">
            <div style="color: #00FF7F; font-size: 2rem; margin-bottom: 20px;">◈</div>
            <h3 style="color: white; font-size: 1.6rem; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">Синтез тех-стратегий</h3>
            <p style="color: #888; font-size: 1.15rem; line-height: 1.7;">
                Математическое объединение лучших практик технического анализа в единый адаптивный алгоритм. Поиск синергии между трендовыми и осцилляторными моделями.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_v3:
    st.markdown("""
        <div style="background: #0A0A0A; padding: 35px; border-radius: 15px; border: 2px solid #222; min-height: 400px; display: flex; flex-direction: column;">
            <div style="color: #4169E1; font-size: 2rem; margin-bottom: 20px;">◈</div>
            <h3 style="color: white; font-size: 1.6rem; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">Фундаментальный фильтр</h3>
            <p style="color: #888; font-size: 1.15rem; line-height: 1.7;">
                Валидация технических сигналов данными финансовой отчетности и мультипликаторами. Исключение входов в «пустые» активы на основе глубокого анализа бизнеса.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- 6. ФИНАЛЬНЫЙ БЛОК: КОНТРАСТНЫЙ ФУТЕР ---

st.markdown('<div style="margin-top: 80px; border-top: 1px solid #333; padding-top: 40px;"></div>', unsafe_allow_html=True)

col_footer_left, col_footer_right = st.columns([1, 2], gap="large")

with col_footer_left:
    st.markdown("""
        <div style="background: #0A0A0A; padding: 25px; border-radius: 12px; border: 1px solid #333; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #FFD700; font-weight: 700; text-transform: uppercase; font-size: 1rem; margin-bottom: 20px; text-align: center; letter-spacing: 1px;">Техническая поддержка</p>
            <a href="https://t.me/your_link" target="_blank" style="text-decoration: none;">
                <div style="background: #222; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: 600; font-size: 1.1rem; border: 1px solid #444;">
                    Связаться с разработчиком
                </div>
            </a>
        </div>
    """, unsafe_allow_html=True)

with col_footer_right:
    st.markdown("""
        <div style="background: #0A0A0A; padding: 25px; border-radius: 12px; border: 1px solid #222; height: 180px; display: flex; align-items: center;">
            <p style="color: #999; font-size: 1rem; line-height: 1.6; text-align: justify; margin: 0;">
                <strong style="color: #CCC; font-size: 1.1rem; letter-spacing: 1px;">DISCLAIMER:</strong> 
                Представленные расчеты являются математической моделью и не гарантируют аналогичный результат в будущем. 
                Алгоритмическая торговля на MOEX сопряжена с рыночными рисками. Инструмент предназначен исключительно 
                для аналитических целей и не является индивидуальной инвестиционной рекомендацией.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Самый нижний копирайт — четкий и контрастный
st.markdown("""
    <div style="margin-top: 50px; padding-bottom: 40px; display: flex; justify-content: flex-end; align-items: center; gap: 15px;">
        <div style="width: 40px; height: 1px; background: #444;"></div>
        <span style="color: #888; font-family: monospace; font-size: 1rem; letter-spacing: 4px; text-transform: uppercase;">
            prod. by <span style="color: #FFF; font-weight: 700;">JiDi©</span>
        </span>
    </div>
""", unsafe_allow_html=True)