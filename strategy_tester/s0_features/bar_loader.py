"""S0.1 Bar Loader — load and cache OHLCV bars for a pair universe.

Loads OHLCV from yfinance and caches to a single parquet with MultiIndex
(ticker, field) columns. No forward-fill — raw data with NaN preserved.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _last_business_day() -> pd.Timestamp:
    """Most recent Monday-Friday that is <= yesterday."""
    yesterday = pd.Timestamp.now().normalize() - timedelta(days=1)
    offset = max(0, yesterday.weekday() - 4)
    return yesterday - timedelta(days=offset)


def _parse_universe(
    universe_path: str | Path,
) -> tuple[list[str], list[dict]]:
    """Extract unique tickers and pair dicts from universe JSON."""
    with open(universe_path) as f:
        data = json.load(f)
    pairs = data["ratios"]
    tickers: set[str] = set()
    for p in pairs:
        tickers.add(p["numerator"])
        tickers.add(p["denominator"])
    return sorted(tickers), pairs


class UniverseBars:
    """Cached OHLCV bars for a full pair universe.

    Wraps a MultiIndex DataFrame with (ticker, field) columns.
    No forward-fill applied — raw data with NaN preserved.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        pairs: list[dict],
        source_map: dict[str, str],
    ) -> None:
        self._bars = bars
        self._pairs = pairs
        self._source_map = source_map

    def close(self) -> pd.DataFrame:
        """Wide DataFrame: DatetimeIndex x tickers (Close prices)."""
        return self._bars.xs("Close", axis=1, level="field")

    def open(self) -> pd.DataFrame:
        """Wide DataFrame: DatetimeIndex x tickers (Open prices)."""
        return self._bars.xs("Open", axis=1, level="field")

    def high(self) -> pd.DataFrame:
        """Wide DataFrame: DatetimeIndex x tickers (High prices)."""
        return self._bars.xs("High", axis=1, level="field")

    def low(self) -> pd.DataFrame:
        """Wide DataFrame: DatetimeIndex x tickers (Low prices)."""
        return self._bars.xs("Low", axis=1, level="field")

    def volume(self) -> pd.DataFrame:
        """Wide DataFrame: DatetimeIndex x tickers (Volume)."""
        return self._bars.xs("Volume", axis=1, level="field")

    def tickers(self) -> list[str]:
        """All unique tickers in the universe."""
        return sorted(
            self._bars.columns.get_level_values("ticker")
            .unique()
            .tolist()
        )

    def pairs(self) -> list[tuple[str, str, str]]:
        """List of (name, numerator, denominator) tuples."""
        return [
            (p["name"], p["numerator"], p["denominator"])
            for p in self._pairs
        ]

    def ratio(self, num: str, den: str) -> pd.Series:
        """Compute close_num / close_den on common dates."""
        close = self.close()
        s_num = close[num]
        s_den = close[den]
        common = s_num.dropna().index.intersection(s_den.dropna().index)
        return (s_num.loc[common] / s_den.loc[common]).rename(f"{num}/{den}")

    def pair_close(
        self, num: str, den: str,
    ) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
        """Return (num_close, den_close, common_index) for a pair."""
        close = self.close()
        s_num = close[num]
        s_den = close[den]
        common = s_num.dropna().index.intersection(s_den.dropna().index)
        return s_num.loc[common], s_den.loc[common], common

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(first_date, last_date) across all tickers."""
        return self._bars.index.min(), self._bars.index.max()

    def coverage(self) -> pd.DataFrame:
        """Per-ticker: first_date, last_date, n_bars, source."""
        close = self.close()
        rows = []
        for ticker in self.tickers():
            series = close[ticker].dropna()
            rows.append({
                "first_date": (
                    series.index.min() if len(series) else pd.NaT
                ),
                "last_date": (
                    series.index.max() if len(series) else pd.NaT
                ),
                "n_bars": len(series),
                "source": self._source_map.get(ticker, "unknown"),
            })
        return pd.DataFrame(rows, index=self.tickers())

    def stale_tickers(self, days: int = 7) -> list[str]:
        """Tickers whose last Close date is > N calendar days old."""
        cutoff = pd.Timestamp.now().normalize() - timedelta(days=days)
        close = self.close()
        stale = []
        for ticker in self.tickers():
            series = close[ticker].dropna()
            if len(series) > 0 and series.index.max() < cutoff:
                stale.append(ticker)
        return stale

    def missing_tickers(self) -> list[str]:
        """Tickers with no data from any source."""
        return [
            t for t, src in self._source_map.items() if src == "missing"
        ]

    def __len__(self) -> int:
        """Number of unique tickers."""
        return len(self.tickers())

    def __repr__(self) -> str:
        first, last = self.date_range()
        return (
            f"UniverseBars({len(self)} tickers, "
            f"{len(self._pairs)} pairs, "
            f"{first.date()} to {last.date()})"
        )


def _fetch_from_yfinance(
    missing_tickers: list[str],
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch OHLCV for missing tickers from yfinance.

    Downloads in batches of 50 to avoid rate limits.
    Uses auto_adjust=False to match MDA unadjusted prices.
    Strips timezone info to match MDA's tz-naive index.
    """
    import yfinance as yf

    BATCH_SIZE = 50
    source_updates: dict[str, str] = {}
    all_frames: list[pd.DataFrame] = []

    for i in range(0, len(missing_tickers), BATCH_SIZE):
        batch = missing_tickers[i : i + BATCH_SIZE]
        logger.info(
            "yfinance batch %d/%d: %d tickers",
            i // BATCH_SIZE + 1,
            (len(missing_tickers) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch),
        )
        try:
            raw = yf.download(
                batch,
                period="max",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            logger.warning("yfinance batch download failed: %s", e)
            for t in batch:
                source_updates[t] = "missing"
            continue

        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        raw = raw.loc[raw.index <= cutoff]

        for ticker in batch:
            try:
                if len(batch) == 1:
                    ticker_data = raw
                else:
                    ticker_data = raw[ticker]

                if ticker_data["Close"].notna().any():
                    cols = {}
                    for field in FIELDS:
                        if field in ticker_data.columns:
                            cols[(ticker, field)] = ticker_data[field]
                        else:
                            cols[(ticker, field)] = np.nan
                    frame = pd.DataFrame(cols, index=ticker_data.index)
                    frame.columns = pd.MultiIndex.from_tuples(
                        frame.columns,
                        names=["ticker", "field"],
                    )
                    all_frames.append(frame)
                    source_updates[ticker] = "yfinance"
                    logger.info("  %s: OK from yfinance", ticker)
                else:
                    source_updates[ticker] = "missing"
                    logger.warning(
                        "  %s: no data from yfinance", ticker
                    )
            except (KeyError, TypeError):
                source_updates[ticker] = "missing"
                logger.warning(
                    "  %s: failed to extract from yfinance", ticker
                )

    if all_frames:
        result = pd.concat(all_frames, axis=1)
    else:
        result = pd.DataFrame()

    return result, source_updates


def _save_cache(
    bars: pd.DataFrame,
    source_map: dict[str, str],
    cache_path: Path,
) -> None:
    """Save bars to parquet with source_map in schema metadata."""
    table = pa.Table.from_pandas(bars)
    metadata = {
        b"fetch_timestamp": datetime.now().isoformat().encode(),
        b"source_map": json.dumps(source_map).encode(),
    }
    existing = table.schema.metadata or {}
    existing.update(metadata)
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, cache_path)
    logger.info(
        "Cache saved: %s (%.1f MB)",
        cache_path,
        cache_path.stat().st_size / 1e6,
    )


def _load_cache(
    cache_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read parquet cache and extract source_map from metadata."""
    table = pq.read_table(cache_path)
    bars = table.to_pandas()
    meta = table.schema.metadata or {}
    source_map = json.loads(meta.get(b"source_map", b"{}"))
    return bars, source_map


def load_universe_bars(
    universe_path: str | Path,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> UniverseBars:
    """Load OHLCV bars for all tickers in a pair universe."""
    universe_path = Path(universe_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "s0_bars.parquet"

    tickers, pairs = _parse_universe(universe_path)
    cutoff = _last_business_day()

    if cache_path.exists() and not force_refresh:
        bars, source_map = _load_cache(cache_path)
        if bars.index.max() >= cutoff:
            logger.info(
                "Cache hit: %s (last date %s >= cutoff %s)",
                cache_path,
                bars.index.max().date(),
                cutoff.date(),
            )
            return UniverseBars(bars, pairs, source_map)
        logger.info(
            "Cache stale: last date %s < cutoff %s, re-fetching",
            bars.index.max().date(),
            cutoff.date(),
        )

    logger.info("Fetching %d tickers from yfinance...", len(tickers))
    bars, source_map = _fetch_from_yfinance(tickers, cutoff)

    # Mark any tickers with no data as missing
    for t in tickers:
        if t not in source_map:
            source_map[t] = "missing"

    _save_cache(bars, source_map, cache_path)

    ub = UniverseBars(bars, pairs, source_map)
    _print_report(ub)
    return ub


def load_close_dict(
    cache_path: str | Path,
) -> dict[str, pd.Series]:
    """Fast load: read parquet cache → dict[ticker → Close Series].

    Designed for multiprocessing workers that need the legacy dict format
    without re-fetching from ArcticDB/yfinance. Skips universe parsing —
    just reads the cached bars.

    Parameters
    ----------
    cache_path : str | Path
        Path to ``s0_bars.parquet`` (written by ``load_universe_bars``).

    Returns
    -------
    dict[str, pd.Series]
        {ticker: Close pd.Series with DatetimeIndex}, NaN rows dropped.
    """
    bars, _ = _load_cache(Path(cache_path))
    close = bars.xs("Close", axis=1, level="field")
    return {t: close[t].dropna() for t in close.columns}


def load_ohlcv_dict(
    cache_path: str | Path,
) -> dict[str, dict[str, pd.Series]]:
    """Fast load: read parquet cache → nested dict[ticker → {field → Series}].

    Returns all OHLCV fields per ticker.  For workers that need H/L for
    proper ATR/ADX computation.

    Parameters
    ----------
    cache_path : str | Path
        Path to ``s0_bars.parquet``.

    Returns
    -------
    dict[str, dict[str, pd.Series]]
        {ticker: {"Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ...}}
    """
    bars, _ = _load_cache(Path(cache_path))
    result: dict[str, dict[str, pd.Series]] = {}
    tickers = sorted(
        bars.columns.get_level_values("ticker").unique().tolist()
    )
    for ticker in tickers:
        result[ticker] = {}
        for field in FIELDS:
            try:
                result[ticker][field] = bars[(ticker, field)].dropna()
            except KeyError:
                pass
    return result


def _print_report(ub: UniverseBars) -> None:
    """Print summary of loaded universe bars."""
    n_yf = sum(1 for v in ub._source_map.values() if v == "yfinance")
    n_miss = sum(
        1 for v in ub._source_map.values() if v == "missing"
    )
    stale = ub.stale_tickers(days=7)
    first, last = ub.date_range()

    print(f"\n{'=' * 40}")
    print("S0.1 Bar Loader Report")
    print(f"{'=' * 40}")
    print(
        f"Loaded {len(ub)} tickers: "
        f"{n_yf} from yfinance, {n_miss} missing"
    )
    print(f"Date range: {first.date()} to {last.date()}")

    if ub.missing_tickers():
        print("\nMISSING (no data from any source):")
        for t in sorted(ub.missing_tickers()):
            print(f"  {t}")

    if stale:
        print("\nSTALE (last date > 7 days ago):")
        cov = ub.coverage()
        today = pd.Timestamp.now().normalize()
        for t in sorted(stale):
            last_dt = cov.loc[t, "last_date"]
            days_ago = (today - last_dt).days
            print(
                f"  {t}: last={last_dt.date()} ({days_ago} days ago)"
            )

    print(f"{'=' * 40}\n")
