"""Weekday gate — refuse non-Friday runs for weekly strategies unless
explicitly forced by the operator.

Two production weekly strategies are gated by this module:
    - tsmom (signal at Friday close → fill Monday open)
    - fred_regime (Friday weekly rebalance)

Failure modes prevented:
- Cron miswiring (signal_generator firing on Mon-Thu) silently emitting
  stale signals against a non-rebalance date.
- Operator running `signal_generator.py` mid-week thinking the system
  is daily-flow.

Force-run protocol (operator override):
    1. `--force` + `--force-reason "..."` on the CLI (non-interactive,
       suitable for ad-hoc backfill / one-off reruns).
    2. `--force` alone in a TTY: prompts for a reason at the terminal.
       Empty reason → refused (no silent overrides).
    3. `--force` alone in non-TTY (cron): refused. Cron must commit to
       a reason via `--force-reason` or the gate stays.

Every accepted force is logged at WARNING level with the reason +
strategy name + actual weekday, so the audit trail is preserved.

Usage
-----
    from strategy_tester.data.weekday_gate import require_friday_or_force

    # at the top of run() / generate_signals() / main():
    require_friday_or_force(
        strategy_name="tsmom",
        force=args.force,
        force_reason=args.force_reason,
        logger=log,
        # optional: target_date=args.date (already-parsed datetime/date)
    )

The function returns None on success or raises SystemExit(1) on refusal.
Callers should NOT catch SystemExit — let it propagate to the CLI.
"""

from __future__ import annotations

import logging
import sys
from datetime import date as date_cls
from datetime import datetime
from typing import Union

# Monday=0 .. Sunday=6 — Friday = 4
FRIDAY = 4
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

DateLike = Union[date_cls, datetime, str, None]


def _coerce_date(d: DateLike) -> date_cls:
    if d is None:
        return datetime.now().date()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def _prompt_for_reason(strategy_name: str, today: date_cls) -> str:
    """Interactive reason prompt. Empty/whitespace input → empty string
    (caller refuses the force). No silent overrides."""
    day = _WEEKDAY_NAMES[today.weekday()]
    sys.stderr.write(
        f"\n⚠️  {strategy_name} is a Friday-only strategy. Today is "
        f"{day} {today.isoformat()}.\n"
    )
    sys.stderr.write(
        "    Provide a reason for the force-run (one line, "
        "logged in WARN-level audit trail).\n"
    )
    sys.stderr.write("    Empty reason → refuses the force.\n")
    sys.stderr.write("Reason: ")
    sys.stderr.flush()
    try:
        reason = sys.stdin.readline().strip()
    except (EOFError, KeyboardInterrupt):
        reason = ""
    return reason


def require_friday_or_force(
    *,
    strategy_name: str,
    force: bool = False,
    force_reason: str | None = None,
    target_date: DateLike = None,
    logger: logging.Logger | None = None,
) -> None:
    """Refuse the run unless today (or `target_date`) is a Friday or the
    operator explicitly forces.

    `target_date` lets the caller pin the gate to the SIGNAL's intended
    date, not the calendar today — so `--date 2026-06-05` (a Friday)
    can be replayed on a Monday without forcing.

    Raises SystemExit(1) on refusal (after writing a clear message to
    stderr / logger).
    """
    log = logger or logging.getLogger(__name__)
    target = _coerce_date(target_date)
    day = _WEEKDAY_NAMES[target.weekday()]

    if target.weekday() == FRIDAY:
        log.info(
            "%s weekday gate: %s %s — OK",
            strategy_name,
            day,
            target.isoformat(),
        )
        return  # Friday — proceed unconditionally

    # Non-Friday path: refuse unless force is explicit
    if not force:
        log.error(
            "%s is a Friday-only strategy. target_date=%s is %s — refusing. "
            'Pass --force --force-reason "..." to override (or run with '
            "--date <a-friday> for a back-dated signal).",
            strategy_name,
            target.isoformat(),
            day,
        )
        sys.exit(1)

    # --force passed: need a non-empty reason from somewhere
    reason = (force_reason or "").strip()
    if not reason and sys.stdin.isatty():
        reason = _prompt_for_reason(strategy_name, target)
    if not reason:
        log.error(
            "%s --force requested on %s but no reason provided "
            "(empty --force-reason and non-TTY or empty prompt input). "
            "Refusing — silent force-runs are blocked by policy.",
            strategy_name,
            day,
        )
        sys.exit(1)

    log.warning(
        "%s WEEKDAY GATE FORCED: today=%s (%s), reason=%r",
        strategy_name,
        target.isoformat(),
        day,
        reason,
    )


def skip_weekend_nav(
    *,
    target_date: DateLike,
    explicit: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    """Return True when the caller should SKIP writing a daily NAV row.

    A NAV mark on a market-closed weekend day just duplicates Friday's flat
    row, so an unattended weekend run (a launcher that crossed midnight into
    Saturday, or a manual weekend session) would append a spurious point to
    the equity curve. Returns False (proceed) on weekdays, or whenever
    `explicit` is set — the operator passed --date, an intentional backfill
    that must always be honoured (including weekend back-dating).
    """
    d = _coerce_date(target_date)
    if explicit or d.weekday() <= FRIDAY:
        return False
    log = logger or logging.getLogger(__name__)
    log.info(
        "NAV skip: %s is a %s (market closed) and no --date was given — "
        "no NAV change to record.",
        d.isoformat(),
        _WEEKDAY_NAMES[d.weekday()],
    )
    return True
