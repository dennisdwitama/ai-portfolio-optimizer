from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import run_backtest, summarize_performance
from .config import TECHNICAL_FEATURES
from .features import build_technical_features
from .data_loader import compute_returns, download_price_data
from .ml_models import cluster_stocks, predict_next_day_scores, prepare_ml_dataset, train_random_forest_models
from .optimizer import optimize_portfolio


def get_price_and_volume_data(tickers: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = download_price_data(tickers=tickers, start=start, end=end)

    volumes = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    try:
        import yfinance as yf

        volume_data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )
        if isinstance(volume_data.columns, pd.MultiIndex) and "Volume" in volume_data.columns.get_level_values(0):
            volumes = volume_data["Volume"].copy().reindex(prices.index)
    except Exception:
        pass

    return prices, volumes


def run_full_analysis(
    tickers: list[str],
    start: str,
    end: str | None,
    risk_target: float,
) -> dict[str, Any]:
    if len(tickers) < 2:
        raise ValueError("Please choose at least two tickers.")

    prices, volumes = get_price_and_volume_data(tickers, start, end)
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Not enough price history was available for the selected assets.")

    returns = compute_returns(prices)
    clusters = cluster_stocks(returns)

    feature_panel = {}
    for ticker in prices.columns:
        volume_series = volumes[ticker] if ticker in volumes.columns else None
        panel = build_technical_features(prices[ticker], volume_series=volume_series)
        feature_panel[ticker] = panel

    dataset = prepare_ml_dataset(feature_panel)
    if dataset.empty or len(dataset) < 50:
        raise ValueError("Not enough feature history was available to train the ML models.")
    regressor, classifier = train_random_forest_models(dataset)

    latest_rows = []
    for ticker, df in feature_panel.items():
        row = df.tail(1).copy()
        row["ticker"] = ticker
        latest_rows.append(row)
    latest_df = pd.concat(latest_rows)
    latest_df = latest_df.dropna(subset=TECHNICAL_FEATURES)
    if latest_df.empty:
        raise ValueError("No recent feature rows were available for prediction.")

    predictions = predict_next_day_scores(latest_df, regressor, classifier)
    ai_alpha = predictions.set_index("ticker")["predicted_return"] * 252

    positive_tickers = predictions.query("prob_up >= 0.5")["ticker"].tolist()
    if len(positive_tickers) < 3:
        positive_tickers = predictions.head(min(6, len(predictions)))["ticker"].tolist()

    optimized_prices = prices[positive_tickers].dropna(axis=1, how="all")
    if optimized_prices.shape[1] < 2:
        optimized_prices = prices.dropna(axis=1, how="all")
    optimized_weights, optimized_perf = optimize_portfolio(optimized_prices, risk_target, ai_alpha=ai_alpha)
    mv_weights, mv_perf = optimize_portfolio(prices, risk_target, ai_alpha=None)

    backtest_returns, weights_history = run_backtest(prices, risk_target=risk_target)
    perf_summary = summarize_performance(backtest_returns) if not backtest_returns.empty else pd.DataFrame()

    return {
        "prices": prices,
        "returns": returns,
        "clusters": clusters,
        "dataset": dataset,
        "regressor": regressor,
        "classifier": classifier,
        "predictions": predictions,
        "optimized_weights": optimized_weights,
        "optimized_performance": optimized_perf,
        "mv_weights": mv_weights,
        "mv_performance": mv_perf,
        "backtest_returns": backtest_returns,
        "weights_history": weights_history,
        "performance_summary": perf_summary,
    }
