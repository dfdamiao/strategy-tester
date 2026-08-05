"""Per-ticker freshness check for signal generators.

Replaces the two duplicate inline ``validate_data_freshness`` functions in
tsmom and obv_pivot signal_generators. Generalised to:

* per-ticker stale flag (vs old all-or-nothing return),
* missing-ticker handling (counts as stale with sentinel -1 days),
* tolerance window (``max_days_behind``) for weekend / non-trading days.

The function does not connect to anything. Pair it with
``strategy_tester.data.ibkr_gap_fill.{prompt_for_fallback,
fill_gaps_from_ibkr}`` to recover stale tickers from IBKR.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StaleReport:
    fresh_tickers: list[str] = field(default_factory=list)
    stale_tickers: list[str] = field(default_factory=list)
    days_behind: dict[str, int] = field(default_factory=dict)
    effective_target_date: dt.date | None = None

    @property
    def is_all_stale(self) -> bool:
        return bool(self.stale_tickers) and not self.fresh_tickers

    def log_summary(self, log: logging.Logger | None = None) -> None:
        log = log or logger
        log.info(
            "freshness: %d fresh, %d stale",
            len(self.fresh_tickers),
            len(self.stale_tickers),
        )
        for t in self.stale_tickers[:8]:
            log.warning("  stale %s: %d days behind", t, self.days_behind.get(t, -1))
        if len(self.stale_tickers) > 8:
            log.warning("  … and %d more stale tickers", len(self.stale_tickers) - 8)


_CALENDAR_CACHE: dict[str, Any] = {}


def _last_session_on_or_before(target: dt.date, calendar: str) -> dt.date:
    """Most recent ``calendar`` session on or before ``target``.

    Rolls a holiday / weekend target back to the last completed session
    (e.g. Fri 2026-07-03, July 4th observed → Thu 2026-07-02). Falls back
    to the raw target on any calendar failure — a freshness check must
    never crash a signal run.
    """
    try:
        cal = _CALENDAR_CACHE.get(calendar)
        if cal is None:
            import exchange_calendars as xcals

            cal = xcals.get_calendar(calendar)
            _CALENDAR_CACHE[calendar] = cal
        return cal.date_to_session(target.isoformat(), direction="previous").date()
    except Exception as exc:
        logger.warning(
            "calendar %s unavailable (%s) — using raw target_date %s",
            calendar,
            exc,
            target,
        )
        return target


def check_freshness(
    prices: dict[str, pd.DataFrame],
    target_date: dt.date,
    *,
    max_days_behind: int = 0,
    expected_tickers: list[str] | None = None,
    calendar: str | None = "XNYS",
) -> StaleReport:
    """Compare each ticker's last bar against ``target_date``.

    Args:
        prices: dict[ticker -> OHLCV DataFrame].
        target_date: the business date the caller is running for.
        max_days_behind: tolerance window. ``0`` (default) means the last bar
            must be on or after ``target_date``. Use ``1`` to tolerate one
            non-trading day (e.g. running before today's close).
        expected_tickers: optional list of tickers the caller expected to
            receive. Tickers in this list but missing from ``prices`` count
            as stale with ``days_behind = -1`` (sentinel for "missing").
        calendar: exchange-calendars name used to roll a non-session
            ``target_date`` back to the last completed session (default
            ``"XNYS"`` — a US-holiday Friday compares against Thursday's
            close instead of marking every US ticker stale, the 2026-07-03
            incident). Pass ``None`` to compare against the raw date.

    Returns:
        StaleReport with sorted fresh / stale lists + per-ticker
        days_behind (measured vs ``effective_target_date``). The lists are
        sorted alphabetically for deterministic logging / testing.
    """
    effective_target = target_date
    if calendar is not None:
        effective_target = _last_session_on_or_before(target_date, calendar)
        if effective_target != target_date:
            logger.info(
                "freshness: %s is not an %s session — comparing against "
                "last session %s",
                target_date,
                calendar,
                effective_target,
            )
    report = StaleReport(effective_target_date=effective_target)
    seen: set[str] = set()
    for t, df in prices.items():
        seen.add(t)
        if df is None or df.empty:
            report.stale_tickers.append(t)
            report.days_behind[t] = -1
            continue
        # A frame can be non-empty yet carry a NaN last Close: yfinance pads
        # each ticker to the bulk-download's union calendar, so a halted /
        # delisted ticker keeps trailing NaN rows at today's date. Drop the
        # NaN tail before reading the last bar, else index[-1] reports the
        # union-max date and the stale ticker is silently marked fresh.
        if "Close" in df.columns:
            df = df[df["Close"].notna()]
            if df.empty:
                report.stale_tickers.append(t)
                report.days_behind[t] = -1
                continue
        last_ts = df.index[-1]
        last_d: dt.date
        if isinstance(last_ts, pd.Timestamp):
            last_d = last_ts.date()
        elif isinstance(last_ts, dt.date):
            last_d = last_ts
        else:
            # Fall back to pandas coercion for numpy datetime64, str, etc.
            last_d = pd.Timestamp(cast(Any, last_ts)).date()  # pyright: ignore[reportAssignmentType]
        days = (effective_target - last_d).days
        report.days_behind[t] = days
        if days <= max_days_behind:
            report.fresh_tickers.append(t)
        else:
            report.stale_tickers.append(t)

    if expected_tickers:
        for t in expected_tickers:
            if t not in seen:
                report.stale_tickers.append(t)
                report.days_behind[t] = -1

    report.fresh_tickers.sort()
    report.stale_tickers.sort()
    return report


def filter_fresh(
    prices: dict[str, pd.DataFrame],
    report: StaleReport,
    *,
    protect: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a new dict containing only the tickers in ``report.fresh_tickers``.

    Args:
        protect: tickers kept even when stale (e.g. currently-held
            positions — a held name must be scored on its last available
            bar, never silently dropped and force-exited).
    """
    keep = set(report.fresh_tickers)
    if protect:
        protected_stale = protect & set(report.stale_tickers) & set(prices)
        if protected_stale:
            logger.warning(
                "filter_fresh: keeping %d stale-but-protected tickers "
                "(held positions): %s",
                len(protected_stale),
                sorted(protected_stale),
            )
        keep |= protected_stale
    return {t: prices[t] for t in sorted(keep) if t in prices}
