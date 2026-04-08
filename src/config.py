DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "NVDA", "TSLA",
    "AMZN", "META", "JPM", "UNH", "XOM"
]

RISK_LEVEL_MAP = {
    "Conservative": 0.12,
    "Balanced": 0.18,
    "Aggressive": 0.28,
}

TECHNICAL_FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_10d",
    "volatility_21d",
    "sma_10_ratio",
    "sma_21_ratio",
    "sma_50_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_width",
    "volume_change_5d",
]
