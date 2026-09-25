"""Retry merge must keep the retried ticker's full history.

The first bulk call can return a ticker empty on a short index (Yahoo batch
hiccup); the per-ticker retry then downloads its full history. Merging the
retry by assigning columns into the first call's frame aligns them onto that
frame's index and silently drops every bar outside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import strategy_tester.data.yf_retry as yr


def test_retry_merge_keeps_retried_history(monkeypatch) -> None:
    fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    idx_first = pd.bdate_range("2010-01-04", periods=10)
    idx_retry = pd.bdate_range("2009-12-21", periods=20)
    first = pd.DataFrame(
        np.nan,
        index=idx_first,
        columns=pd.MultiIndex.from_product([fields, ["AAA", "BBB"]]),
    )
    for f in fields:
        first[(f, "AAA")] = 1.0
    retry = pd.DataFrame(
        2.0,
        index=idx_retry,
        columns=pd.MultiIndex.from_product([fields, ["BBB"]]),
    )
    calls: list[list[str]] = []

    def fake_download(tickers, **_kw):
        calls.append(list(tickers))
        return first.copy() if len(calls) == 1 else retry.copy()

    monkeypatch.setattr(yr.yf, "download", fake_download)
    monkeypatch.setattr(yr.time, "sleep", lambda _s: None)

    raw, failed = yr.download_with_retry(["AAA", "BBB"], period="max", max_retries=1)

    assert failed == [], failed
    bbb = raw[("Close", "BBB")].dropna()
    assert bbb.index.min() == idx_retry[0] and len(bbb) == len(idx_retry), (
        f"after the retry, BBB has {len(bbb)} bar(s) starting "
        f"{bbb.index.min().date()}; expected the retry's full "
        f"{len(idx_retry)}-bar history from {idx_retry.min().date()}"
    )
    aaa = raw[("Close", "AAA")].dropna()
    assert len(aaa) == len(idx_first), "AAA must keep its own bars"
