"""Tests for `weekday_gate.require_friday_or_force`.

Covers all 5 decision paths:
    1. Friday — passes unconditionally
    2. Non-Friday + no --force → SystemExit(1)
    3. Non-Friday + --force + --force-reason → logs WARN, returns
    4. Non-Friday + --force + empty reason + non-TTY → SystemExit(1)
    5. target_date Friday on a non-Friday calendar day → passes
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime

import pytest

from strategy_tester.data.weekday_gate import (
    require_friday_or_force,
    skip_weekend_nav,
)

FRIDAY = date(2026, 6, 5)
MONDAY = date(2026, 6, 1)
SATURDAY = date(2026, 6, 6)  # the 2026-06-06 spurious-NAV-row incident date
SUNDAY = date(2026, 6, 7)


def test_friday_passes(caplog):
    """Friday calendar day → returns None, no exit."""
    with caplog.at_level(logging.INFO):
        require_friday_or_force(
            strategy_name="tsmom",
            target_date=FRIDAY,
        )
    assert any("OK" in r.message for r in caplog.records)


def test_non_friday_refuses():
    """Monday + no force → SystemExit(1)."""
    with pytest.raises(SystemExit) as exc:
        require_friday_or_force(
            strategy_name="tsmom",
            target_date=MONDAY,
        )
    assert exc.value.code == 1


def test_force_with_explicit_reason_passes(caplog):
    """Monday + --force + --force-reason → logs WARN with reason."""
    with caplog.at_level(logging.WARNING):
        require_friday_or_force(
            strategy_name="fred_regime",
            target_date=MONDAY,
            force=True,
            force_reason="ad-hoc rerun after CSV repair",
        )
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("FORCED" in m and "rerun" in m for m in msgs)


def test_force_empty_reason_non_tty_refuses(monkeypatch):
    """--force passed but no reason in a non-TTY context (e.g. cron):
    must refuse — silent overrides are blocked by policy."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as exc:
        require_friday_or_force(
            strategy_name="tsmom",
            target_date=MONDAY,
            force=True,
            force_reason="",
        )
    assert exc.value.code == 1


def _fake_tty_stdin(text: str) -> io.StringIO:
    """Build a StringIO that pretends to be a TTY (isatty() == True)."""
    fake = io.StringIO(text)
    fake.isatty = lambda: True  # type: ignore[assignment]
    return fake


def test_force_tty_prompt_accepts_reason(monkeypatch, caplog):
    """--force in a TTY context: prompt is rendered, reason from stdin
    is accepted and logged."""
    monkeypatch.setattr(
        "sys.stdin",
        _fake_tty_stdin("backfill for missed Friday run\n"),
    )
    with caplog.at_level(logging.WARNING):
        require_friday_or_force(
            strategy_name="tsmom",
            target_date=MONDAY,
            force=True,
        )
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("backfill" in m for m in msgs)


def test_force_tty_prompt_empty_refuses(monkeypatch):
    """--force in a TTY but user hits Enter on prompt → refuses (empty
    reason still counts as silent)."""
    monkeypatch.setattr("sys.stdin", _fake_tty_stdin("\n"))
    with pytest.raises(SystemExit) as exc:
        require_friday_or_force(
            strategy_name="tsmom",
            target_date=MONDAY,
            force=True,
        )
    assert exc.value.code == 1


def test_target_date_friday_overrides_calendar_today():
    """--date 2026-06-05 (Friday) replayed on Monday: gate passes because
    the SIGNAL date is Friday, not the calendar day."""
    require_friday_or_force(
        strategy_name="tsmom",
        target_date=FRIDAY,  # caller passes the signal date
        force=False,
    )  # should NOT raise


def test_datetime_target_coerces_to_date():
    """datetime input is coerced to date — Friday-datetime passes."""
    require_friday_or_force(
        strategy_name="tsmom",
        target_date=datetime(2026, 6, 5, 16, 30),
    )  # should NOT raise


# --- skip_weekend_nav (daily NAV writers refuse weekend rows) -------------


def test_skip_weekend_nav_saturday_skips(caplog):
    """Saturday + no explicit date → skip (True) and log the reason."""
    with caplog.at_level(logging.INFO):
        assert skip_weekend_nav(target_date=SATURDAY) is True
    assert any("market closed" in r.message for r in caplog.records)


def test_skip_weekend_nav_sunday_skips():
    assert skip_weekend_nav(target_date=SUNDAY) is True


def test_skip_weekend_nav_weekday_proceeds():
    """Mon-Fri → do not skip (False)."""
    assert skip_weekend_nav(target_date=FRIDAY) is False
    assert skip_weekend_nav(target_date=date(2026, 6, 8)) is False  # Monday


def test_skip_weekend_nav_explicit_date_honoured():
    """Operator passed --date on a weekend (intentional backfill) → proceed."""
    assert skip_weekend_nav(target_date=SATURDAY, explicit=True) is False


def test_skip_weekend_nav_accepts_str_and_datetime():
    """Date string and datetime both coerce; weekend still skips, weekday not."""
    assert skip_weekend_nav(target_date="2026-06-06") is True
    assert skip_weekend_nav(target_date=datetime(2026, 6, 6, 18, 17)) is True
    assert skip_weekend_nav(target_date="2026-06-08") is False
