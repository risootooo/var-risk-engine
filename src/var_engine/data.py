"""Data layer: price download, caching and portfolio return construction.

Design notes
------------
* Free data only: Yahoo Finance via ``yfinance``, cached to CSV so every run
  after the first is fully offline and reproducible.
* THE COLUMN-ORDER GUARD: ``yfinance`` returns price columns *alphabetically*,
  not in request order.  Every frame is reindexed to the configured ticker
  order immediately after download; ``ensure_order`` is unit-tested because
  attaching weights in alphabetical order silently corrupts the portfolio
  (this exact bug was caught during verification of the v1 engine).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TICKERS: list[str] = ["SPY", "QQQ", "TLT", "LQD", "GLD"]
WEIGHTS: np.ndarray = np.array([0.35, 0.20, 0.20, 0.10, 0.15])
DEFAULT_START = "2007-01-01"


def ensure_order(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return ``frame`` with columns exactly in ``tickers`` order.

    Raises if a requested ticker is missing (a data problem must be loud,
    never silently filled).
    """
    missing = [t for t in tickers if t not in frame.columns]
    if missing:
        raise ValueError(f"Missing tickers in price frame: {missing}")
    return frame[tickers]


def load_prices(
    cache_path: str | Path,
    tickers: list[str] = TICKERS,
    start: str = DEFAULT_START,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load adjusted daily prices, from cache when available."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        log.info("Loaded cached prices: %s rows (%s -> %s)", len(prices), prices.index.min().date(), prices.index.max().date())
        return ensure_order(prices, tickers)

    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):  # single-ticker edge case
        prices = prices.to_frame(tickers[0])
    prices = ensure_order(prices.dropna(how="all"), tickers).dropna()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    log.info("Downloaded prices: %s rows (%s -> %s)", len(prices), prices.index.min().date(), prices.index.max().date())
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; the whole study is in log-return space (documented)."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray, tickers: list[str] = TICKERS) -> pd.Series:
    """Fixed-weight, daily-rebalanced portfolio log returns.

    Daily rebalancing of fixed weights is a documented simplification;
    weights are validated to sum to 1 so misuse fails fast.
    """
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {w.sum():.4f}")
    ordered = ensure_order(asset_returns, tickers)
    port = ordered.values @ w
    return pd.Series(port, index=ordered.index, name="portfolio")
