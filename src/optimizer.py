from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    from pypfopt.exceptions import OptimizationError
except Exception:  # pragma: no cover - exercised in dependency-constrained environments
    EfficientFrontier = None
    expected_returns = None
    risk_models = None

    class OptimizationError(Exception):
        pass


def _normalize_with_caps(raw_weights: pd.Series, max_weight: float = 0.35) -> pd.Series:
    weights = raw_weights.clip(lower=0).fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=raw_weights.index)
    weights = weights / weights.sum()

    for _ in range(len(weights) * 2):
        over_cap = weights > max_weight
        if not over_cap.any():
            break

        capped = weights.where(~over_cap, max_weight)
        residual = 1.0 - capped.sum()
        under_cap = ~over_cap
        if residual <= 0 or not under_cap.any():
            weights = capped / capped.sum()
            break

        under_values = capped[under_cap]
        if under_values.sum() <= 0:
            capped.loc[under_cap] = residual / under_cap.sum()
        else:
            capped.loc[under_cap] = under_values / under_values.sum() * residual
        weights = capped

    return weights / weights.sum()


def _fallback_optimize(
    price_window: pd.DataFrame,
    risk_target: float,
    ai_alpha: pd.Series | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    returns = price_window.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Not enough data to optimize the portfolio.")

    mu = returns.mean() * 252
    if ai_alpha is not None:
        ai_alpha = ai_alpha.reindex(mu.index).fillna(0)
        mu = 0.7 * mu + 0.3 * ai_alpha

    cov = returns.cov() * 252
    ridge = np.eye(len(cov)) * 1e-6
    inv_cov = np.linalg.pinv(cov.to_numpy() + ridge)
    risk_aversion = max(risk_target**2, 1e-4)
    scores = pd.Series(inv_cov @ mu.to_numpy(), index=mu.index) / risk_aversion
    weights = _normalize_with_caps(scores)

    portfolio_return = float(weights.dot(mu))
    portfolio_vol = float(np.sqrt(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy()))
    sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else np.nan
    return weights.round(6).to_dict(), {
        "expected_return": portfolio_return,
        "volatility": portfolio_vol,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
    }


def optimize_portfolio(
    price_window: pd.DataFrame,
    risk_target: float,
    ai_alpha: pd.Series | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    cleaned = price_window.dropna(axis=1, how="all").ffill().dropna()
    if cleaned.shape[1] < 2:
        raise ValueError("At least two assets with clean price history are required for optimization.")

    if EfficientFrontier is None or expected_returns is None or risk_models is None:
        return _fallback_optimize(cleaned, risk_target, ai_alpha)

    mu = expected_returns.mean_historical_return(cleaned, frequency=252)
    if ai_alpha is not None:
        ai_alpha = ai_alpha.reindex(mu.index).fillna(0)
        mu = 0.7 * mu + 0.3 * ai_alpha

    cov = risk_models.sample_cov(cleaned, frequency=252)
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 0.35))

    try:
        ef.efficient_risk(target_volatility=risk_target)
    except (ValueError, OptimizationError):
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, 0.35))
        ef.max_sharpe()

    weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
    summary = {
        "expected_return": float(performance[0]),
        "volatility": float(performance[1]),
        "sharpe": float(performance[2]),
    }
    return weights, summary


def equal_weight_portfolio(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    if n == 0:
        raise ValueError("Cannot build an equal-weight portfolio with no tickers.")
    return {ticker: 1 / n for ticker in tickers}


def weights_to_series(weights: dict[str, float], columns: list[str]) -> pd.Series:
    return pd.Series(weights).reindex(columns).fillna(0.0)
