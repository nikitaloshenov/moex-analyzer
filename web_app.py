import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    st.markdown("""
    <style>
    .loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }

    .loader {
        width: 60px;
        height: 60px;
        border: 5px solid rgba(255,255,255,0.15);
        border-top: 5px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        100% { transform: rotate(360deg); }
    }

    .loader-text {
        margin-top: 20px;
        color: white;
        font-size: 18px;
        font-family: sans-serif;
        letter-spacing: 1px;
    }
    </style>

    <div class="loader-overlay">
        <div class="loader"></div>
        <div class="loader-text">Run analysis...</div>
    </div>
    """, unsafe_allow_html=True)

class QuantDataProvider:
    if not st.session_state.loaded:
        time.sleep(5)
    """Класс-абстракция для подготовки и генерации данных моделей."""

    @staticmethod
    def get_equity_curve_data() -> pd.DataFrame:
        """Возвращает данные для плавной восходящей кривой капитала."""
        x = np.linspace(0, 10, 100)
        y = 100 + 15 * x + np.sin(x) * 2
        return pd.DataFrame({"Time": x, "Equity": y})

    @staticmethod
    def get_pipeline_parameters() -> list[dict]:
        """Возвращает параметры для карточек пайплайна."""
        return [
            {"title": "Целевой объём исполнения", "value": "150"},
            {"title": "Коэффициент импакта", "value": "3.2"},
            {"title": "Оценка оптимального AUM", "value": "900k"}
        ]

    @staticmethod
    def get_candlestick_data(n=100, start_price=100) -> pd.DataFrame:
        np.random.seed(42)

        time = np.arange(n)

        # базовая случайная доходность
        returns = np.random.randn(n) * 0.5

        price = start_price + np.cumsum(returns)

        open_ = price + np.random.randn(n) * 0.2
        close = price + np.random.randn(n) * 0.2

        high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)

        df = pd.DataFrame({
            "Time": time,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close
        })

        return df

    @staticmethod
    def get_backtest_market_impact_data() -> pd.DataFrame:
        """Возвращает данные для графиков PnL Net, PnL Mid и Market Impact."""
        x = np.linspace(0, 50, 100)

        return pd.DataFrame({
            "Time": x,
            "PnL Net": 100 - x * 1 + np.random.randn(100) * 3,
            "PnL Mid": 100 + x * 0.7 + np.random.randn(100) * 1,
            "Market Impact": x * 0.3 + np.random.randn(100) * 2
        })

    @staticmethod
    def get_aum_optimization_data() -> pd.DataFrame:
        """Возвращает данные для оценки оптимального AUM."""
        x = np.linspace(10, 1000, 50)
        y = -0.0001 * (x - 500) ** 2 + 100
        return pd.DataFrame({"AUM": x, "Efficiency": y})

    @staticmethod
    @st.cache_data
    def load_svg(file_path: str) -> str:
        """Вспомогательный метод для безопасного чтения SVG-файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return """
            < svg
            xmlns = "http://w3.org"
            viewBox = "0 0 200 60"
            width = "100%"
            height = "100%" >
            < polygon
            points = "0,0 170,0 200,30 170,60 0,60 30,30"
            fill = "#262730"
            stroke = "#FFD700"
            stroke - width = "2" / >
        < / svg >"""


class QuantPlatformApp:
    """Класс интерфейса платформы, отвечающий за рендеринг блоков макета."""

    custom_divider = lambda: st.markdown("""
<div style="
    width: 100%;
    height: 2px;
    background-color: #333344;
    margin: 25px 0;
"></div>
""", unsafe_allow_html=True)

    def __init__(self):
        # Инициализируем провайдер данных внутри приложения
        self.data_provider = QuantDataProvider()

        # Настройка страницы (вынесена в конструктор)
        st.set_page_config(
            page_title="MOEX Quant Execution Platform",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def render_header(self):
        """Заголовок платформы."""

        title = "MOEX Quant Execution Platform"
        caption = "Платформа анализа рынка и симуляции ордеров"
        st.title(title)
        st.caption(caption)

        QuantPlatformApp.custom_divider()

    def render_equity_section(self):
        """БЛОК 1: Плавная восходящая кривая капитала. !!!НЕ ИСПОЛЬЗУЕТСЯ!!!"""

        subh = "Плавная восходящая кривая капитала"
        equity_data = self.data_provider.get_equity_curve_data()
        desc = "Результат стратегии является выигрышным за счет..."

        st.subheader(subh)
        st.line_chart(equity_data, x="Time", y="Equity", color="#FFD700")
        st.info(desc)

        QuantPlatformApp.custom_divider()

    def render_pipeline_arrow_section(self):
        """Отрисовывает 5 блоков-стрелок в линию с вашим текстом внутри."""
        st.subheader("Обзор логики системы (Pipeline)")
        svg_path: str = r".\web\.images\Frame 3.svg"

        # Читаем файл фоновой SVG-фигуры
        svg_content = QuantDataProvider.load_svg(svg_path)

        st.markdown(svg_content, unsafe_allow_html=True)

        QuantPlatformApp.custom_divider()

        with st.expander("Параметры системы", expanded=True):
            cards = self.data_provider.get_pipeline_parameters()
            cards_table = ([], [], [])
            for i in range(len(cards)):
                cards_table[i % 3].append(cards[i])

            cols = st.columns(3)

            with cols[0]:
                for card in cards_table[0]:
                    st.metric(label=card["title"], value=card["value"])
            with cols[1]:
                for card in cards_table[1]:
                    st.metric(label=card["title"], value=card["value"])
            with cols[2]:
                for card in cards_table[2]:
                    st.metric(label=card["title"], value=card["value"])
        QuantPlatformApp.custom_divider()

    def render_signal_section(self):
        """БЛОК 3: M2: Сигнал."""

        subh = "M2: Сигнал"
        caption = "Отображение сигнальных точек входа и выхода на свечном графике на основе анализа рыночного дисбаланса, ценовой динамики и краткосрочных паттернов поведения рынка. Система формирует торговые сигналы для последующего бэктеста и оценки качества исполнения."
        candle_data = self.data_provider.get_candlestick_data()

        st.subheader(subh)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=candle_data["Time"],
            open=candle_data["Open"],
            high=candle_data["High"],
            low=candle_data["Low"],
            close=candle_data["Close"],
            name="Market"
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,  # убирает нижний слайдер
            hovermode="x unified"
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.caption(caption)

        x_min = candle_data["Time"].min()
        x_max = candle_data["Time"].max()

        y_min = candle_data["Low"].min()
        y_max = candle_data["High"].max()

        y_padding = (y_max - y_min) * 0.1

        fig.update_layout(
            xaxis=dict(
                range=[x_min, x_max]
            ),
            yaxis=dict(
                range=[
                    y_min - y_padding,
                    y_max + y_padding
                ]
            )
        )

        QuantPlatformApp.custom_divider()

    def render_backtest_section(self):
        """БЛОК 4: M3 + M4: Backtest + Market Impact."""

        subh = "M3 + M4: Backtest + Market Impact"


        st.subheader(subh)
        col_desc, col_charts = st.columns(2)

        with col_desc:
            st.markdown("### M3: Бектест")
            st.write("Метрики исторического моделирования торгового алгоритма с оценкой доходности, стабильности и риска стратегии на исторических рыночных данных. Отображаются ключевые показатели производительности, включая динамику PnL, винрейт и коэффициенты риск/доходность.")
            st.markdown("### M4: Market ")
            st.write("Анализ проскальзывания и влияния объёма исполнения на рыночную цену. Модель оценивает стоимость исполнения ордеров, деградацию качества сделок и расхождение между теоретическим и фактическим результатом торговли.")

        with col_charts:
            st.markdown("#### График PnL Net + PnL Mid + Market Impact")

            df = self.data_provider.get_backtest_market_impact_data()

            fig = go.Figure()

            # 🔵 PnL Mid (верхняя линия)
            fig.add_trace(go.Scatter(
                x=df["Time"],
                y=df["PnL Mid"],
                mode="lines",
                name="PnL Mid",
                line=dict(color="#20E080", width=2)
            ))

            # 🟡 PnL Net (нижняя линия + заливка до Mid)
            fig.add_trace(go.Scatter(
                x=df["Time"],
                y=df["PnL Net"],
                mode="lines",
                name="PnL Net",
                line=dict(color="#ADADDD", width=2),
                fill="tonexty",
                fillcolor="rgba(240,32,128,0.125)"  # 🔴 Impact зона
            ))

            # ⚙️ оформление как терминал
            fig.update_layout(
                title="PnL Mid vs PnL Net (Market Impact)",
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title="PnL",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False}
            )

            x_min = df["Time"].min()
            x_max = df["Time"].max()

            y_min = df["PnL Net"].min()
            y_max = df["PnL Mid"].max()

            y_padding = (y_max - y_min) * 0.1

            fig.update_layout(
                xaxis=dict(
                    range=[x_min, x_max]
                ),
                yaxis=dict(
                    range=[
                        y_min - y_padding,
                        y_max + y_padding
                    ]
                )
            )

        QuantPlatformApp.custom_divider()

    def render_aum_section(self):
        """БЛОК 5: Оценка оптимального AUM."""
        st.subheader("Оценка оптимального AUM")
        col_aum_desc, col_aum_chart = st.columns(2)

        with col_aum_desc:
            st.markdown("### Оптимальный AUM")
            st.write("Анализ зависимости чистой доходности стратегии от объёма капитала под управлением (AUM). Точка перегиба кривой показывает предельную ёмкость стратегии, при которой дальнейшее увеличение объёма начинает снижать эффективность из-за роста рыночного импакта и издержек исполнения.")

        with col_aum_chart:
            aum_data = self.data_provider.get_aum_optimization_data()

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=aum_data["AUM"],
                y=aum_data["Efficiency"],
                mode="lines",
                name="Efficiency",
                line=dict(color="#FF5733", width=3)
            ))

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="AUM",
                yaxis_title="Efficiency",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

    def run(self):
        if not st.session_state.loaded:
            st.session_state.loaded = True
            st.rerun()
        """Основной метод для последовательной сборки интерфейса."""
        self.render_header()
        self.render_pipeline_arrow_section()
        # self.render_equity_section()
        self.render_signal_section()
        self.render_backtest_section()
        self.render_aum_section()


# Точка входа в приложение Streamlit
if __name__ == "__main__":
    app = QuantPlatformApp()
    app.run()

