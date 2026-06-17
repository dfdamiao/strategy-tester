"""yfinance bulk download with per-ticker retry on transient failures.

Yahoo's bulk endpoint occasionally returns NaN/None for a single ticker even
when the rest of the batch succeeds (typical symptom: TSMOM 2026-05-05 dry-run
where XLI returned `'NoneType' object is not subscriptable`). This helper
re-downloads only the failed subset up to `max_retries` times with
exponential backoff before giving up.

Use as a drop-in replacement for `yfinance.download(...)`:

    from strategy_tester.data.yf_retry import download_with_retry
    raw, failed = download_with_retry(tickers, period="2y", auto_adjust=False)

Returns the same DataFrame shape `yf.download` would return, plus a list
of tickers that still failed after all retries (caller decides handling —
existing freshness gates already drop NaN columns).
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubles per attempt → 1, 2, 4


def _failed_tickers(raw: pd.DataFrame | None, tickers: list[str]) -> list[str]:
    """Tickers that are missing from `raw` or whose Close column is all-NaN."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return list(tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return list(tickers)
        close = raw["Close"]
    else:
        # Single-ticker: columns are field names, no per-ticker level
        if "Close" not in raw.columns:
            return list(tickers)
        close = raw[["Close"]].copy()
        close.columns = pd.Index(tickers[:1])
    failed: list[str] = []
    for t in tickers:
        if t not in close.columns:
            failed.append(t)
            continue
        if bool(pd.isna(close[t]).all()):
            failed.append(t)
    return failed


def _merge_retry(
    raw: pd.DataFrame | None,
    retry_raw: pd.DataFrame | None,
    retried: list[str],
) -> None:
    """Merge a retry result back into `raw` in place."""
    if raw is None or retry_raw is None or retry_raw.empty:
        return
    if isinstance(retry_raw.columns, pd.MultiIndex):
        for col in retry_raw.columns:
            field, ticker = col[0], col[1]
            if ticker not in retried:
                continue
            raw[(field, ticker)] = retry_raw[col]
    else:
        # Single-ticker retry
        ticker = retried[0]
        for field in retry_raw.columns:
            raw[(field, ticker)] = retry_raw[field]


def download_with_retry(
    tickers: list[str],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **yf_kwargs,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Wrap yf.download with per-ticker retry on detected failures.

    Args:
        tickers: list of ticker symbols.
        max_retries: number of retry attempts after the initial bulk call.
        backoff_base: starting sleep duration; doubles each attempt.
        **yf_kwargs: forwarded to yf.download (period, auto_adjust, ...).

    Returns:
        (raw_df, permanently_failed_tickers)
    """
    if not tickers:
        return pd.DataFrame(), []

    # Log every initial attempt — previously only retries+failures were visible.
    _period = yf_kwargs.get("period")
    _start = yf_kwargs.get("start")
    _end = yf_kwargs.get("end")
    _interval = yf_kwargs.get("interval", "1d")
    _range = f"period={_period}" if _period else f"start={_start} end={_end}"
    logger.info(
        f"yf.download: {len(tickers)} ticker(s), {_range}, interval={_interval}, "
        f"auto_adjust={yf_kwargs.get('auto_adjust')}, "
        f"progress={yf_kwargs.get('progress', False)}"
    )
    t0 = time.time()
    raw = yf.download(tickers, **yf_kwargs)
    failed = _failed_tickers(raw, tickers)
    dt_s = time.time() - t0
    if not failed:
        logger.info(
            f"yf.download OK: {len(tickers)} ticker(s) in {dt_s:.1f}s"
        )
        return raw, []
    logger.warning(
        f"yf.download partial: {len(tickers) - len(failed)}/{len(tickers)} OK "
        f"in {dt_s:.1f}s, {len(failed)} failed → retrying"
    )

    for attempt in range(1, max_retries + 1):
        sleep_s = backoff_base * (2 ** (attempt - 1))
        logger.info(
            f"yf retry {attempt}/{max_retries} for {len(failed)} ticker(s) "
            f"(sleep {sleep_s:.0f}s): {failed}"
        )
        time.sleep(sleep_s)
        retry_raw = yf.download(failed, **yf_kwargs)
        _merge_retry(raw, retry_raw, failed)
        failed = _failed_tickers(raw, tickers)
        if not failed:
            logger.info(
                f"yf.download recovered all tickers after retry {attempt}"
            )
            return raw, []

    logger.warning(
        f"yf gave up: {len(failed)} ticker(s) still failed after "
        f"{max_retries} retries: {failed}"
    )
    return raw, failed
