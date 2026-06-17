"""Unit tests for strategy_tester.s2_signal.hl_spread_band.

Hand-traced 30-bar synthetic OHLC fixture:

    bars 0-14  : close = 100, high = 100.5, low = 99.5    (stable, builds HH)
    bars 15-16 : close = 90,  high = 91,    low = 89      (sharp drop = entry)
    bar  17    : close = 102, high = 103,   low = 101     (exit-by-high-cross)
    bars 18-29 : close = 102, high = 102.5, low = 101.5

With k=2.0, N=10, M=5:
    band(t=15) = max(close[5:15]) - 2.0 * mean(spread[10:15])
               = 100.0 - 2.0 * 1.0
               = 98.0
    close[15] = 90 < 98 → entry fires at bar 15 (and bar 16 by the same logic).

    exit(t=17): close[17]=102 > high[16]=91 → high-cross exit fires at bar 17.

    regime(t=15, R=5): mean(close[11:16]) = (100+100+100+100+90)/5 = 98.
                       close[15]=90 < 98 → regime exit fires at bar 15
                       (only present when R > 0).
"""
from __future__ import annotations

import numpy as np
import pytest

from strategy_tester.s2_signal.hl_spread_band import compute_hl_spread_band

pytestmark = pytest.mark.unit


@pytest.fixture
def ohlc_30bar() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """30-bar OHLC fixture per module docstring."""
    close = np.array(
        [100.0] * 15
        + [90.0, 90.0]
        + [102.0] * 13,
        dtype=np.float64,
    )
    assert len(close) == 30
    high = close + 0.5
    low = close - 0.5
    return close, high, low


def test_entry_fires_on_band_break(ohlc_30bar) -> None:
    """Bar 15: close=90 drops below band=98 → entry True; nowhere else."""
    close, high, low = ohlc_30bar
    entry, _ = compute_hl_spread_band(close, high, low, 2.0, 10, 5, 0)
    assert entry[15]
    assert entry[16]
    entry_days = np.flatnonzero(entry)
    assert set(entry_days.tolist()) == {15, 16}


def test_exit_by_high_cross(ohlc_30bar) -> None:
    """Bar 17: close=102 > high[16]=91 → high-cross exit True."""
    close, high, low = ohlc_30bar
    _, exit_signal = compute_hl_spread_band(close, high, low, 2.0, 10, 5, 0)
    assert exit_signal[17]


def test_exit_by_regime_only_with_R_enabled(ohlc_30bar) -> None:
    """Bar 15: regime SMA(R=5)=96 > close=90 → regime exit fires only when R>0.

    With R=0 the regime branch is disabled and bar 15 has no high-cross
    (close[15]=90 ≤ high[14]=96.5), so exit[15] must be False.
    """
    close, high, low = ohlc_30bar
    _, exit_r0 = compute_hl_spread_band(close, high, low, 2.0, 10, 5, 0)
    _, exit_r5 = compute_hl_spread_band(close, high, low, 2.0, 10, 5, 5)
    assert not exit_r0[15]
    assert exit_r5[15]
    assert exit_r5.sum() >= exit_r0.sum()


def test_warmup_no_signals_in_first_N_bars(ohlc_30bar) -> None:
    """No entries possible before i=max(N,M)=10; verified on stable prefix."""
    close, high, low = ohlc_30bar
    entry, _ = compute_hl_spread_band(close, high, low, 2.0, 10, 5, 0)
    assert entry[:10].sum() == 0
