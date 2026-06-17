"""Shared test fixtures for paper-trade signal_generator integration tests."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest


def _fake_ohlcv(ticker: str, last_date: dt.date, n_bars: int = 750) -> pd.DataFrame:
    """Deterministic synthetic OHLCV — ~3y of business days ending on last_date."""
    idx = pd.date_range(end=last_date, periods=n_bars, freq="B")
    base = 100 + hash(ticker) % 50
    closes = [base + 0.1 * i for i in range(n_bars)]
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n_bars,
        },
        index=idx,
    )


@pytest.fixture
def fresh_today():
    """Today's date for synthetic data freshness."""
    return dt.date.today()


@pytest.fixture
def mock_get_prices(fresh_today, monkeypatch):
    """Replace lib.data.get_prices with a deterministic fake."""

    def _fake(tickers, period, target_date, **kw):
        return {t: _fake_ohlcv(t, fresh_today) for t in tickers}

    import strategy_tester.data as ldata
    monkeypatch.setattr(ldata, "get_prices", _fake)
    from strategy_tester.data import yf_bulk_cache
    monkeypatch.setattr(yf_bulk_cache, "get_prices", _fake)
    return _fake
