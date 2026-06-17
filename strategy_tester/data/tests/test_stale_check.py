"""Tests for strategy_tester/data/stale_check.py."""
from __future__ import annotations

import datetime as dt

import pandas as pd


def _frame(last_date: dt.date) -> pd.DataFrame:
    idx = pd.date_range(end=last_date, periods=5, freq="B")
    return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)


def test_all_fresh_returns_empty_stale_list():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    prices = {"SPY": _frame(target), "QQQ": _frame(target)}
    report = check_freshness(prices, target_date=target)
    assert report.fresh_tickers == ["QQQ", "SPY"]
    assert report.stale_tickers == []
    assert report.is_all_stale is False


def test_all_stale_flips_is_all_stale():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    stale_day = dt.date(2026, 5, 20)
    prices = {"SPY": _frame(stale_day), "QQQ": _frame(stale_day)}
    report = check_freshness(prices, target_date=target)
    assert report.fresh_tickers == []
    assert set(report.stale_tickers) == {"SPY", "QQQ"}
    assert report.is_all_stale is True
    assert report.days_behind["SPY"] > 0


def test_mixed_freshness():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    prices = {
        "SPY": _frame(target),
        "QQQ": _frame(dt.date(2026, 5, 22)),
        "IWM": _frame(target),
    }
    report = check_freshness(prices, target_date=target)
    assert report.fresh_tickers == ["IWM", "SPY"]
    assert report.stale_tickers == ["QQQ"]
    assert report.days_behind["QQQ"] == (target - dt.date(2026, 5, 22)).days
    assert report.is_all_stale is False


def test_missing_ticker_counts_as_stale():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    prices = {"SPY": _frame(target)}  # QQQ entirely missing
    report = check_freshness(
        prices, target_date=target, expected_tickers=["SPY", "QQQ"],
    )
    assert "QQQ" in report.stale_tickers
    assert report.days_behind["QQQ"] == -1  # sentinel for missing


def test_max_days_behind_tolerance():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    one_day_back = dt.date(2026, 5, 25)
    prices = {"SPY": _frame(one_day_back)}
    report = check_freshness(prices, target_date=target, max_days_behind=1)
    assert report.stale_tickers == []
    assert report.fresh_tickers == ["SPY"]


def test_empty_input_returns_empty_report():
    from strategy_tester.data.stale_check import check_freshness

    target = dt.date(2026, 5, 26)
    report = check_freshness({}, target_date=target)
    assert report.fresh_tickers == []
    assert report.stale_tickers == []
    assert report.is_all_stale is False  # vacuously not stale


def test_filter_prices_drops_stale():
    from strategy_tester.data.stale_check import check_freshness, filter_fresh

    target = dt.date(2026, 5, 26)
    prices = {
        "SPY": _frame(target),
        "QQQ": _frame(dt.date(2026, 5, 22)),
    }
    report = check_freshness(prices, target_date=target)
    fresh_only = filter_fresh(prices, report)
    assert set(fresh_only.keys()) == {"SPY"}
