"""Tests for strategy_tester/data/yf_bulk_cache.py.

Uses a monkeypatched download_with_retry so tests do not touch the network.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def fake_ohlcv():
    """Build a deterministic 3-bar OHLCV frame keyed by date."""
    idx = pd.date_range("2026-05-22", periods=3, freq="B", tz="UTC")
    return pd.DataFrame(
        {"Open": [100, 101, 102], "High": [101, 102, 103],
         "Low": [99, 100, 101], "Close": [100.5, 101.5, 102.5],
         "Volume": [1_000, 1_100, 1_200]},
        index=idx,
    )


@pytest.fixture
def patched_download(fake_ohlcv, monkeypatch):
    """Patch download_with_retry to return a deterministic dict + failed list."""
    from strategy_tester.data import yf_bulk_cache as cache_mod

    mock = MagicMock()

    def _fake(tickers, period, **kw):
        frames = {}
        for t in tickers:
            for col in ("Open", "High", "Low", "Close", "Volume"):
                frames[(col, t)] = fake_ohlcv[col].values
        idx = fake_ohlcv.index
        df = pd.DataFrame(frames, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        mock.calls.append((tuple(sorted(tickers)), period))
        return df, []

    mock.calls = []
    monkeypatch.setattr(cache_mod, "download_with_retry", _fake)
    return mock


def test_cache_miss_writes_parquet(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    target = dt.date(2026, 5, 26)
    prices = get_prices(
        tickers=["SPY", "QQQ"], period="2y", target_date=target,
        cache_dir=tmp_path,
    )
    assert set(prices.keys()) == {"SPY", "QQQ"}
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1
    assert len(patched_download.calls) == 1


def test_cache_hit_skips_download(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    target = dt.date(2026, 5, 26)
    _ = get_prices(["SPY", "QQQ"], "2y", target, cache_dir=tmp_path)
    _ = get_prices(["SPY", "QQQ"], "2y", target, cache_dir=tmp_path)
    # Second call must hit cache, so only one download recorded.
    assert len(patched_download.calls) == 1


def test_cache_per_day_ttl(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    _ = get_prices(["SPY"], "2y", dt.date(2026, 5, 26), cache_dir=tmp_path)
    _ = get_prices(["SPY"], "2y", dt.date(2026, 5, 27), cache_dir=tmp_path)
    # Different target_date → different cache key → second call redownloads.
    assert len(patched_download.calls) == 2


def test_refresh_bypasses_cache(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    target = dt.date(2026, 5, 26)
    _ = get_prices(["SPY"], "2y", target, cache_dir=tmp_path)
    _ = get_prices(["SPY"], "2y", target, cache_dir=tmp_path, refresh=True)
    assert len(patched_download.calls) == 2


def test_subset_of_cached_tickers_returns_subset(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    target = dt.date(2026, 5, 26)
    _ = get_prices(["SPY", "QQQ", "IWM"], "2y", target, cache_dir=tmp_path)
    prices = get_prices(["SPY"], "2y", target, cache_dir=tmp_path)
    # Cache key uses sorted-tickers tuple, so requesting subset triggers a new
    # download (we don't slice cached results; identical query only).
    assert "SPY" in prices
    assert len(patched_download.calls) == 2


def test_returns_per_ticker_dict_shape(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import get_prices

    target = dt.date(2026, 5, 26)
    prices = get_prices(["SPY", "QQQ"], "2y", target, cache_dir=tmp_path)
    assert isinstance(prices["SPY"], pd.DataFrame)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(prices["SPY"].columns)


def test_clear_cache_removes_files(tmp_path, patched_download):
    from strategy_tester.data.yf_bulk_cache import clear_cache, get_prices

    target = dt.date(2026, 5, 26)
    _ = get_prices(["SPY"], "2y", target, cache_dir=tmp_path)
    removed = clear_cache(cache_dir=tmp_path)
    assert removed == 1
    assert list(tmp_path.glob("*.parquet")) == []
