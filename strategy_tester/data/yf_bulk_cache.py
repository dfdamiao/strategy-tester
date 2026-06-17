"""Shared yfinance bulk cache.

Single fetch point for the paper-trade signal generators. The first call on
a given (sorted-tickers, period, target_date) tuple performs a bulk download
via ``download_with_retry`` and writes the resulting per-ticker OHLCV frames
to a single parquet file at ``cache/yf_bulk/<hash>.parquet`` (stacked by
ticker as the outer index level). Subsequent calls in the same day load
from disk so the 8 strategies share one fetch.

Cache TTL: per-day, encoded in the cache key. There is no time-based
invalidation; if you need a fresh pull on the same day, pass ``refresh=True``.

Format: parquet (per project convention for DataFrames).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

from strategy_tester.data.yf_retry import download_with_retry

logger = logging.getLogger(__name__)

# Cache lives under the current working directory by default; override by
# passing cache_dir= to get_prices(), or set YF_BULK_CACHE_DIR.
DEFAULT_CACHE_DIR = Path(
    os.environ.get("YF_BULK_CACHE_DIR", "cache/yf_bulk")
)


def _cache_key(
    tickers: list[str], period: str, target_date: dt.date, *, auto_adjust: bool
) -> str:
    # auto_adjust must be in the key: a split/dividend-adjusted frame and a raw
    # frame for the same (tickers, period, date) are different data. Omitting it
    # lets a caller silently read another strategy's wrong-adjustment prices.
    payload = (
        f"{period}|{target_date.isoformat()}|aa={int(auto_adjust)}|"
        f"{','.join(sorted(tickers))}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
_STACKED_COLS = ["date", "ticker", *_OHLCV_COLS]


def _split_multiindex_to_dict(
    raw: pd.DataFrame | None, tickers: list[str]
) -> dict[str, pd.DataFrame]:
    """Convert yfinance MultiIndex output (field, ticker) -> dict[ticker -> df]."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            cols: dict[str, pd.Series] = {}
            for field in _OHLCV_COLS:
                if (field, t) in raw.columns:
                    cols[field] = pd.Series(raw[(field, t)])
            if cols:
                out[t] = pd.DataFrame(cols)
    else:
        # Single-ticker download — columns are field names already.
        if tickers:
            out[tickers[0]] = pd.DataFrame(raw[_OHLCV_COLS]).copy()
    return out


def _dict_to_stacked_frame(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack dict[ticker->df] into a single DataFrame with `ticker` column."""
    if not prices:
        return pd.DataFrame(columns=pd.Index(_STACKED_COLS))
    parts: list[pd.DataFrame] = []
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        copy = df.copy()
        copy["ticker"] = t
        copy = copy.reset_index().rename(columns={"index": "date"})
        if "Date" in copy.columns:
            copy = copy.rename(columns={"Date": "date"})
        parts.append(copy)
    if not parts:
        return pd.DataFrame(columns=pd.Index(_STACKED_COLS))
    return pd.concat(parts, ignore_index=True)


def _stacked_frame_to_dict(stacked: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Inverse of `_dict_to_stacked_frame`."""
    out: dict[str, pd.DataFrame] = {}
    if stacked is None or stacked.empty:
        return out
    for t, sub in stacked.groupby("ticker"):
        df = sub.drop(columns=["ticker"]).set_index("date").sort_index()
        out[str(t)] = pd.DataFrame(df[_OHLCV_COLS])
    return out


def get_prices(
    tickers: list[str],
    period: str,
    target_date: dt.date,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    auto_adjust: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return per-ticker OHLCV frames, served from disk cache when possible."""
    if not tickers:
        return {}
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(tickers, period, target_date, auto_adjust=auto_adjust)
    cache_path = cdir / f"{key}.parquet"

    if cache_path.exists() and not refresh:
        try:
            stacked = pd.read_parquet(cache_path)
            cache_mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            logger.info(
                "yf_bulk_cache HIT  %s tickers=%d period=%s target=%s "
                "(file=%s, written=%s, %.1f KB)",
                key, len(tickers), period, target_date,
                cache_path.name, cache_mtime.strftime("%H:%M:%S"),
                cache_path.stat().st_size / 1024,
            )
            return _stacked_frame_to_dict(stacked)
        except Exception as exc:
            logger.warning(
                "yf_bulk_cache load failed for %s: %s — redownloading", key, exc
            )

    logger.info(
        "yf_bulk_cache MISS %s tickers=%d period=%s target=%s — "
        "calling yf.download(progress=True, auto_adjust=%s, threads=True)",
        key, len(tickers), period, target_date, auto_adjust,
    )
    raw, failed = download_with_retry(
        tickers, period=period, auto_adjust=auto_adjust,
        threads=True, progress=True,
    )
    if failed:
        logger.warning(
            "yfinance permanent failures (%d of %d tickers): %s",
            len(failed), len(tickers), failed,
        )
    prices = _split_multiindex_to_dict(raw, tickers)
    if prices:
        sample_t = next(iter(prices))
        sample_df = prices[sample_t]
        if not sample_df.empty:
            logger.info(
                "yf_bulk_cache download OK: %d/%d tickers populated, "
                "bars %s..%s (sample=%s, n=%d)",
                len(prices), len(tickers),
                str(sample_df.index.min())[:10],
                str(sample_df.index.max())[:10],
                sample_t, len(sample_df),
            )
    # Only persist a COMPLETE, non-empty pull. Writing a partial (some tickers
    # failed) or empty frame poisons every later same-day run on this key —
    # the failed tickers would never be re-fetched (no intra-day TTL), and a
    # pre-close / outage run would lock the whole roster into bad data.
    if prices and not failed:
        try:
            stacked = _dict_to_stacked_frame(prices)
            stacked.to_parquet(cache_path, index=False)
        except Exception as exc:
            logger.warning("yf_bulk_cache write failed for %s: %s", key, exc)
    elif failed:
        logger.warning(
            "yf_bulk_cache NOT cached for %s (%d ticker(s) failed) — next "
            "same-day run will re-download to retry the failures.",
            key, len(failed),
        )
    return prices


def clear_cache(cache_dir: Path | None = None) -> int:
    """Remove all cache files. Returns count of files removed."""
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    if not cdir.exists():
        return 0
    count = 0
    for p in cdir.glob("*.parquet"):
        p.unlink()
        count += 1
    return count
