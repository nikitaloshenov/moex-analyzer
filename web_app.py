from __future__ import annotations

import streamlit as st

from app.services.pipeline import Pipeline, Pypeline


def main() -> None:
    st.set_page_config(
        page_title="MOEX Analyzer",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("MOEX Analyzer")
    st.success("backend connected")
    st.caption("Pipeline and Pypeline are available for the frontend integration.")

    if st.button("Run backend smoke preview"):
        result = Pipeline.run(
            ticker="SBER",
            bars=300,
            execution_mode="optimal",
            dynamic_order_sizing=True,
            target_participation_rate=0.01,
        )
        metrics = result["metrics"]
        st.write(
            "SBER smoke: "
            f"pnl_net={metrics.get('pnl_net', 0.0):.2f}, "
            f"shortfall_ratio={metrics.get('shortfall_ratio', 0.0):.3f}, "
            f"validation_ok={metrics.get('formal_validation_ok', False)}"
        )


if __name__ == "__main__":
    main()
