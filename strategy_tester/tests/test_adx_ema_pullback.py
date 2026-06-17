"""Unit tests for strategy_tester.s2_signal.adx_ema_pullback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s2_signal import adx_ema_pullback as ap


pytestmark = pytest.mark.unit


def _make_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_ema_matches_pandas_reference() -> None:
    """Internal _ema must match pd.ewm(span=N, adjust=False) reference."""
    values = [100.0 + i * 0.3 for i in range(60)]
    s = _make_series(values)
    period = 20
    expected = s.ewm(span=period, adjust=False, min_periods=period).mean()
    actual = ap._ema(s, period)
    pd.testing.assert_series_equal(actual, expected)


def test_pullback_recross_v_shape() -> None:
    """A clear V across the EMA: ratio dips below then crosses back above
    within the lookback window → fires at least one trigger.

    Use a slow uptrend so the EMA tracks closely to price, then a sharp dip
    that actually pierces the EMA, then a recovery above EMA.
    """
    slow_up = [100.0 + i * 0.05 for i in range(40)]   # EMA(20) ~101
    dip = [slow_up[-1] - i * 0.5 for i in range(5)]   # price ~99.5 (below EMA)
    recover = [dip[-1] + i * 0.6 for i in range(15)]
    s = _make_series(slow_up + dip + recover)
    ema = ap._ema(s, period=20)
    # Sanity: confirm the data actually contains a crossing
    crossed_below = (s <= ema).any()
    crossed_back_above = (s.iloc[-10:] > ema.iloc[-10:]).any()
    assert crossed_below and crossed_back_above, (
        "test fixture failed to create a real EMA cross — adjust amplitudes"
    )
    triggers = ap._pullback_recross(s, ema, lookback=5)
    assert triggers.sum() >= 1, "expected at least one pullback recross"


def test_pullback_recross_no_trigger_in_strict_uptrend() -> None:
    """A monotonic uptrend never dips below EMA → zero triggers."""
    s = _make_series([100.0 + i * 0.5 for i in range(60)])
    ema = ap._ema(s, period=20)
    triggers = ap._pullback_recross(s, ema, lookback=5)
    assert triggers.sum() == 0


def test_swing_high_exit_detects_3_up_then_lower() -> None:
    """3 consecutive higher closes followed by a lower close → exit fires
    on the lower-close bar."""
    # Build: HH, HH, HH, then LH (the lower high is what we want to detect)
    base = [100.0, 101.0, 102.0, 103.0, 104.0, 103.5]
    # Pad with leading neutral bars so swing_lookback fully populates
    pad = [100.0] * 5
    s = _make_series(pad + base)
    exits = ap._swing_high_exit(s, swing_lookback=3)
    # The LH is at index pad+5 (the "103.5" bar)
    assert exits.iloc[len(pad) + 5]


def test_swing_high_exit_no_signal_in_choppy_data() -> None:
    """Random walk without sustained 3 consecutive HH should be uncommon.

    The 1-2-3 swing pattern (3 strict HHs then a LH) has expected probability
    ~ 0.5^3 ≈ 0.125 per bar in a coin-flip random walk. Use a lower bound
    (15%) as a sanity check, not a tight discriminator.
    """
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 0.3, size=200).cumsum() + 100.0
    s = _make_series(list(noise))
    exits = ap._swing_high_exit(s, swing_lookback=3)
    assert exits.sum() < len(s) * 0.15


def test_no_entries_when_adx_low_throughout() -> None:
    """A flat-ish series keeps ADX low → no/few entries pass the ADX gate.

    Random noise can occasionally spike ADX, so we accept a small budget
    rather than asserting strict zero.
    """
    rng = np.random.default_rng(seed=7)
    flat = 100.0 + rng.normal(0.0, 0.05, size=80).cumsum()
    s = _make_series(list(flat))
    entries, _ = ap.adx_ema_pullback(
        s, window=20, entry_thresh=50.0,  # very strict ADX bar
    )
    assert entries.sum() <= 1, f"expected <= 1 entry, got {int(entries.sum())}"


def test_strong_uptrend_with_dip_fires_entry() -> None:
    """Slow controlled uptrend so EMA tracks closely → small dip pierces EMA →
    recross fires at least one entry. Weak ADX threshold keeps the gate open.
    """
    slow_up = [100.0 + i * 0.05 for i in range(60)]
    dip = [slow_up[-1] - i * 0.5 for i in range(5)]
    recover = [dip[-1] + i * 0.6 for i in range(20)]
    s = _make_series(slow_up + dip + recover)
    entries, _ = ap.adx_ema_pullback(
        s, window=20, entry_thresh=15.0,  # very mild ADX threshold
    )
    assert entries.sum() >= 1, (
        f"expected at least one entry, got {int(entries.sum())}"
    )


def test_apply_thresholds_no_lookahead() -> None:
    """Both entries and exits are shifted(1) — index 0 must be False."""
    rng = np.random.default_rng(seed=3)
    s = _make_series(list(100.0 + rng.normal(0.5, 0.5, size=80).cumsum()))
    entries, exits = ap.adx_ema_pullback(
        s, window=20, entry_thresh=25.0,
    )
    assert entries.iloc[0] == False  # noqa: E712
    assert exits.iloc[0] == False  # noqa: E712
