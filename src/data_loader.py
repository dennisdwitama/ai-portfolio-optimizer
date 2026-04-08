from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(tickers: Iterable[str], start: str, end: str | None) -> Path:
    suffix = "latest" if end is None else end
    name = f"{'_'.join(sorted(tickers))}_{start}_{suffix}.parquet".replace(":", "-")
    return CACHE_DIR / name


def download_price_data(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices with a simple local cache.

    Notes:
        yfinance relies on Yahoo's public endpoints and may occasionally be rate-limited
        or unavailable. This helper caches successful downloads to make the app more robust.
    """
    cache_file = _cache_path(tickers, start, end)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    try:
        import yfinance as yf
    except Exception as exc:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise RuntimeError(
            "Unable to import yfinance for price downloads. "
            "Install project dependencies and ensure your Python SSL setup works."
        ) from exc

    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise RuntimeError(
            "Price download failed. Check network access and your local SSL configuration."
        ) from exc

    if data.empty:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise ValueError("No price data returned by yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=0).copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()
    prices.to_parquet(cache_file)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
