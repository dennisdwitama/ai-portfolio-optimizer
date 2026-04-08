from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def build_technical_features(price_series: pd.Series, volume_series: pd.Series | None = None) -> pd.DataFrame:
    df = pd.DataFrame(index=price_series.index)
    df["close"] = price_series
    df["return_1d"] = price_series.pct_change(1)
    df["return_5d"] = price_series.pct_change(5)
    df["return_10d"] = price_series.pct_change(10)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_21d"] = df["return_1d"].rolling(21).std()

    sma10 = price_series.rolling(10).mean()
    sma21 = price_series.rolling(21).mean()
    sma50 = price_series.rolling(50).mean()
    df["sma_10_ratio"] = price_series / sma10 - 1
    df["sma_21_ratio"] = price_series / sma21 - 1
    df["sma_50_ratio"] = price_series / sma50 - 1

    df["rsi_14"] = compute_rsi(price_series, 14)
    macd, signal = compute_macd(price_series)
    df["macd"] = macd
    df["macd_signal"] = signal

    rolling_mean = price_series.rolling(20).mean()
    rolling_std = price_series.rolling(20).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    df["bb_width"] = (upper - lower) / rolling_mean

    if volume_series is None:
        volume_series = pd.Series(index=price_series.index, data=1.0)
    df["volume_change_5d"] = volume_series.pct_change(5).replace([np.inf, -np.inf], np.nan)

    df["target_next_return"] = df["return_1d"].shift(-1)
    df["target_up"] = (df["target_next_return"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)
