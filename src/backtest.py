from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TECHNICAL_FEATURES
from .features import build_technical_features
from .ml_models import predict_next_day_scores, prepare_ml_dataset, train_random_forest_models
from .optimizer import equal_weight_portfolio, optimize_portfolio, weights_to_series


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str = "M") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.to_series().groupby(index.to_period(frequency)).tail(1))


def build_feature_panel(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panel = {}
    for ticker in prices.columns:
        panel[ticker] = build_technical_features(prices[ticker])
    return panel


def run_backtest(
    prices: pd.DataFrame,
    risk_target: float,
    rebalance_frequency: str = "M",
    lookback_days: int = 252,
    training_window_days: int = 504,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change().dropna()
    rebalance_dates = _rebalance_dates(returns.index, rebalance_frequency)
    results = []
    weights_log = []

    for rebalance_date in rebalance_dates:
        rebalance_loc = returns.index.get_loc(rebalance_date)
        if rebalance_loc < max(lookback_days, training_window_days):
            continue

        train_slice = prices.iloc[rebalance_loc - training_window_days : rebalance_loc + 1]
        opt_slice = prices.iloc[rebalance_loc - lookback_days : rebalance_loc + 1]

        feature_panel = build_feature_panel(train_slice)
        dataset = prepare_ml_dataset(feature_panel)
        if len(dataset) < 200:
            continue

        regressor, classifier = train_random_forest_models(dataset)

        latest_rows = []
        for ticker, df in feature_panel.items():
            row = df.loc[:rebalance_date].tail(1).copy()
            row["ticker"] = ticker
            latest_rows.append(row)
        latest_feature_rows = pd.concat(latest_rows)
        latest_feature_rows = latest_feature_rows.dropna(subset=TECHNICAL_FEATURES)

        ai_scores = predict_next_day_scores(latest_feature_rows, regressor, classifier)
        ai_alpha = ai_scores.set_index("ticker")["predicted_return"] * 252
        selected_tickers = ai_scores.query("prob_up >= 0.5")["ticker"].tolist()
        if len(selected_tickers) < 3:
            selected_tickers = ai_scores.head(min(6, len(ai_scores)))["ticker"].tolist()

        sub_opt_slice = opt_slice[selected_tickers].dropna(axis=1)
        selected_tickers = sub_opt_slice.columns.tolist()
        if len(selected_tickers) < 2 or opt_slice.dropna(axis=1).shape[1] < 2:
            continue

        try:
            ai_weights, _ = optimize_portfolio(sub_opt_slice, risk_target=risk_target, ai_alpha=ai_alpha)
            mv_weights, _ = optimize_portfolio(opt_slice, risk_target=risk_target, ai_alpha=None)
        except ValueError:
            continue
        ew_weights = equal_weight_portfolio(prices.columns.tolist())

        next_month_returns = returns.loc[rebalance_date:].iloc[1:22]
        if next_month_returns.empty:
            continue

        ai_series = weights_to_series(ai_weights, next_month_returns.columns)
        mv_series = weights_to_series(mv_weights, next_month_returns.columns)
        ew_series = weights_to_series(ew_weights, next_month_returns.columns)

        portfolio_path = pd.DataFrame(index=next_month_returns.index)
        portfolio_path["AI_Optimized"] = (next_month_returns * ai_series).sum(axis=1)
        portfolio_path["MeanVariance"] = (next_month_returns * mv_series).sum(axis=1)
        portfolio_path["EqualWeight"] = (next_month_returns * ew_series).sum(axis=1)
        results.append(portfolio_path)

        weights_frame = pd.DataFrame(
            {
                "AI_Optimized": ai_series,
                "MeanVariance": mv_series,
                "EqualWeight": ew_series,
                "rebalance_date": rebalance_date,
            }
        )
        weights_log.append(weights_frame.reset_index().rename(columns={"index": "ticker"}))

    backtest_returns = pd.concat(results).sort_index() if results else pd.DataFrame()
    weights_history = pd.concat(weights_log).reset_index(drop=True) if weights_log else pd.DataFrame()
    return backtest_returns, weights_history


def summarize_performance(backtest_returns: pd.DataFrame) -> pd.DataFrame:
    summary = {}
    for strategy in backtest_returns.columns:
        series = backtest_returns[strategy].dropna()
        cumulative = (1 + series).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(series)) - 1
        annual_vol = series.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else np.nan
        drawdown = cumulative / cumulative.cummax() - 1
        summary[strategy] = {
            "Total Return": total_return,
            "Annualized Return": annual_return,
            "Annualized Volatility": annual_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown.min(),
        }
    return pd.DataFrame(summary).T
