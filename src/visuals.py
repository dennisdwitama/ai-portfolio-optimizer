from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_price_history(prices: pd.DataFrame):
    normalized = prices / prices.iloc[0]
    fig = px.line(normalized, x=normalized.index, y=normalized.columns, title="Normalized Price History")
    fig.update_layout(legend_title_text="Ticker", xaxis_title="Date", yaxis_title="Normalized Value")
    return fig


def plot_cluster_map(cluster_df: pd.DataFrame):
    fig = px.scatter(
        cluster_df,
        x="annual_vol",
        y="annual_return",
        color="risk_group",
        hover_name=cluster_df.index,
        size="sharpe",
        title="K-Means Stock Risk Clusters",
    )
    fig.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")
    return fig


def plot_cumulative_returns(backtest_returns: pd.DataFrame):
    cumulative = (1 + backtest_returns).cumprod()
    fig = px.line(cumulative, x=cumulative.index, y=cumulative.columns, title="Backtested Strategy Performance")
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value")
    return fig


def plot_weights(weights: dict[str, float], title: str = "Portfolio Weights"):
    series = pd.Series(weights).sort_values(ascending=False)
    fig = px.bar(series, x=series.index, y=series.values, title=title)
    fig.update_layout(xaxis_title="Ticker", yaxis_title="Weight")
    return fig


def plot_feature_importance(model, feature_names: list[str]):
    rf = model.named_steps["rf"]
    importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
    fig.update_layout(title="Random Forest Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
    return fig
