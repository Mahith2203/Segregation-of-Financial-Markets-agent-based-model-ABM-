"""
Real market data loader
========================
Downloads daily adjusted-close prices for a basket of S&P 500 stocks via
`yfinance`, caches them to disk as CSV, and returns log returns.

If `yfinance`/network access is unavailable (e.g. CI, offline dev), falls
back to a clearly-labelled synthetic dataset with realistic sector factor
structure, so the rest of the pipeline (network viz, RMT analysis) still
runs end-to-end for demonstration purposes.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# A representative basket of 40+ S&P 500 large-caps spanning 5 broad GICS-style
# sectors, so results are directly comparable to the ABM's 5-sector design.
SP500_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO", "ADBE", "CRM",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "SPGI",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB", "KMI",
    ],
    "ConsumerDiscretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
    ],
}


def flat_tickers() -> list[str]:
    return [t for tickers in SP500_TICKERS.values() for t in tickers]


def ticker_sector_map() -> dict[str, str]:
    return {t: sec for sec, tickers in SP500_TICKERS.items() for t in tickers}


def fetch_sp500_returns(
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    cache_path: str = "data/sp500_prices.csv",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Returns (log_returns_df, used_real_data).

    log_returns_df: DataFrame indexed by date, columns = tickers.
    used_real_data:  False if we had to fall back to synthetic data.
    """
    if not force_refresh and os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        return returns, True

    try:
        import yfinance as yf

        tickers = flat_tickers()
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        # yfinance returns a MultiIndex column frame when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if prices.shape[1] < 10 or prices.shape[0] < 50:
            raise RuntimeError("Downloaded data looks incomplete.")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        return returns, True

    except Exception as e:  # noqa: BLE001 - deliberately broad: any network/parse failure
        print(f"[data] Could not download live data via yfinance ({e}).")
        print("[data] Falling back to a synthetic 'real-like' dataset so the "
              "pipeline still runs end-to-end. Run again with internet access "
              "for genuine S&P 500 data.")
        returns = _synthetic_market_returns()
        return returns, False


def _synthetic_market_returns(n_days: int = 1000, seed: int = 7) -> pd.DataFrame:
    """A clearly-labelled stand-in dataset with realistic factor structure
    (market factor + sector factors + idiosyncratic noise), used only when
    live data cannot be reached."""
    rng = np.random.default_rng(seed)
    tickers = flat_tickers()
    sectors = list(SP500_TICKERS.keys())
    sector_of = ticker_sector_map()

    dates = pd.bdate_range("2020-01-01", periods=n_days)
    market = rng.normal(0.0003, 0.011, size=n_days)
    sector_factors = {s: rng.normal(0, 0.007, size=n_days) for s in sectors}

    data = {}
    for t in tickers:
        beta = rng.uniform(0.7, 1.3)
        idio = rng.normal(0, 0.012, size=n_days)
        data[t] = beta * market + 0.8 * sector_factors[sector_of[t]] + idio

    return pd.DataFrame(data, index=dates)
