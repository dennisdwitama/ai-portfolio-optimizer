from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.config import DEFAULT_TICKERS, RISK_LEVEL_MAP, TECHNICAL_FEATURES
from src.pipeline import run_full_analysis
from src.visuals import (
    plot_cluster_map,
    plot_cumulative_returns,
    plot_feature_importance,
    plot_price_history,
    plot_weights,
)

st.set_page_config(
    page_title="AI-Driven Portfolio Optimizer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("AI-Driven Portfolio Optimizer")
st.markdown(
    "Build AI-assisted portfolios using **yfinance**, **K-Means**, **Random Forest**, "
    "**mean-variance optimization**, and **5-year backtesting**."
)

with st.sidebar:
    st.header("Controls")
    tickers = st.multiselect(
        "Select stocks",
        options=DEFAULT_TICKERS,
        default=DEFAULT_TICKERS[:6],
    )
    risk_label = st.select_slider(
        "Risk profile",
        options=list(RISK_LEVEL_MAP.keys()),
        value="Balanced",
    )
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=365 * 5))
    end_date = st.date_input("End date", value=date.today())
    run_button = st.button("Run Analysis", type="primary")

if run_button:
    if len(tickers) < 4:
        st.warning("Please select at least 4 stocks so clustering and optimization are meaningful.")
    elif start_date >= end_date:
        st.warning("Please choose an end date after the start date.")
    else:
        try:
            with st.spinner("Fetching data, training models, and running backtests..."):
                results = run_full_analysis(
                    tickers=tickers,
                    start=str(start_date),
                    end=str(end_date),
                    risk_target=RISK_LEVEL_MAP[risk_label],
                )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "If you are running locally, verify that your environment has working "
                "`yfinance`/network access and all dependencies from `requirements.txt` installed."
            )
        else:
            prices = results["prices"]
            clusters = results["clusters"]
            predictions = results["predictions"]
            perf_summary = results["performance_summary"]
            backtest_returns = results["backtest_returns"]
            optimized_weights = results["optimized_weights"]
            mv_weights = results["mv_weights"]
            regressor = results["regressor"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Assets Selected", len(tickers))
            col2.metric("AI Portfolio Holdings", sum(1 for weight in optimized_weights.values() if weight > 0))
            col3.metric("Risk Profile", risk_label)

            st.subheader("Price History")
            st.plotly_chart(plot_price_history(prices), use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.subheader("K-Means Risk Clusters")
                st.plotly_chart(plot_cluster_map(clusters), use_container_width=True)
                st.dataframe(clusters)
            with right:
                st.subheader("Random Forest Predictions")
                st.dataframe(
                    predictions.style.format(
                        {
                            "predicted_return": "{:.4%}",
                            "prob_up": "{:.2%}",
                            "ai_score": "{:.4f}",
                        }
                    )
                )
                st.plotly_chart(
                    plot_feature_importance(regressor, TECHNICAL_FEATURES),
                    use_container_width=True,
                )

            left, right = st.columns(2)
            with left:
                st.subheader("AI-Optimized Weights")
                st.plotly_chart(
                    plot_weights(optimized_weights, "AI-Optimized Portfolio Weights"),
                    use_container_width=True,
                )
                st.json(optimized_weights)
            with right:
                st.subheader("Classical Mean-Variance Weights")
                st.plotly_chart(
                    plot_weights(mv_weights, "Classical Mean-Variance Weights"),
                    use_container_width=True,
                )
                st.json(mv_weights)

            st.subheader("5-Year Backtest")
            if backtest_returns.empty:
                st.info("Backtest could not be completed with the current date range. Try a longer window.")
            else:
                st.plotly_chart(plot_cumulative_returns(backtest_returns), use_container_width=True)
                st.dataframe(
                    perf_summary.style.format(
                        {
                            "Total Return": "{:.2%}",
                            "Annualized Return": "{:.2%}",
                            "Annualized Volatility": "{:.2%}",
                            "Sharpe Ratio": "{:.2f}",
                            "Max Drawdown": "{:.2%}",
                        }
                    )
                )
else:
    st.info("Select stocks and risk level from the sidebar, then click **Run Analysis**.")
