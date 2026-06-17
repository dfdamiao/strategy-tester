"""Tests for the ratio-aware commission model (2026-05-18).

`commission_per_side()` accepts `is_ratio` and `leg_price`. For per-share
models (`ibkr_pro_fixed`, `ibkr_pro_tiered`), `is_ratio=True` triggers a
single-leg (numerator-only) formula whose effective share count is
``notional / leg_price`` — not the walker's synthetic share count, which
balloons for sub-$1 synthetic prices and would otherwise hit the 1% cap.
When `leg_price` is missing, the formula falls back to ``AVG_ETF_PRICE``
($50). `flat_10bps` and `ibkr_lite` are unchanged.

Reference: PORTFOLIO_CONSTRUCTION.md §5.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s5_replay.walker import (
    AVG_ETF_PRICE,
    CASH_BUFFER,
    COST_PER_SIDE,
    IBKR_FIXED_MAX_PCT,
    IBKR_FIXED_MIN_ORDER,
    IBKR_FIXED_PER_SHARE,
    IBKR_TIERED_MIN_ORDER,
    commission_per_side,
    walk_portfolio_oracle,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Unit-level: commission_per_side(is_ratio=..., leg_price=...)
# ---------------------------------------------------------------------------


def test_ratio_fixed_uses_numerator_leg_price() -> None:
    """Bug repro: synth_price=$0.50, $20K notional → 40,000 synthetic shares.

    Without the fix, `ibkr_pro_fixed` hits the 1% cap → $200/side. With
    `is_ratio=True` + `leg_price=$50`, real shares = 20000/50 = 400 →
    max($1, 400 × $0.005) = $2/side, single leg.
    """
    shares = 40_000           # synthetic share count
    notional = 20_000.0
    buggy = commission_per_side(shares, notional, model="ibkr_pro_fixed")
    fixed = commission_per_side(
        shares, notional, model="ibkr_pro_fixed",
        is_ratio=True, leg_price=50.0,
    )
    # Buggy state: 1% cap binds (1% of $20K = $200)
    assert buggy == pytest.approx(200.0)
    # Ratio-aware: max($1, 400 × $0.005) = $2.00 (single leg)
    expected = max(IBKR_FIXED_MIN_ORDER, (20000 / 50) * IBKR_FIXED_PER_SHARE)
    assert fixed == pytest.approx(expected)
    assert 1.0 <= fixed <= 5.0


def test_ratio_fixed_high_priced_etf_hits_minimum() -> None:
    """For a $250 numerator (SPY-tier), $20K notional → 80 shares.
    Per-share = 80 × $0.005 = $0.40, below the $1 min → min binds.
    """
    out = commission_per_side(
        shares=40_000, notional=20_000.0,
        model="ibkr_pro_fixed", is_ratio=True, leg_price=250.0,
    )
    assert out == pytest.approx(IBKR_FIXED_MIN_ORDER)


def test_ratio_fixed_falls_back_to_avg_etf_price_when_leg_price_missing() -> None:
    """`leg_price=0.0` (or omitted) triggers AVG_ETF_PRICE fallback."""
    explicit = commission_per_side(
        shares=40_000, notional=20_000.0,
        model="ibkr_pro_fixed", is_ratio=True, leg_price=AVG_ETF_PRICE,
    )
    fallback = commission_per_side(
        shares=40_000, notional=20_000.0,
        model="ibkr_pro_fixed", is_ratio=True,
    )
    assert explicit == fallback


def test_ratio_tiered_uses_leg_price() -> None:
    """Tiered ratio: leg_shares = 20K/$50 = 400, base = max($0.35, 400×$0.0035) = $1.40,
    passthrough = 400 × $0.0008 = $0.32, total = $1.72 (single leg).
    """
    out = commission_per_side(
        shares=40_000, notional=20_000.0,
        model="ibkr_pro_tiered", is_ratio=True, leg_price=50.0,
    )
    assert out == pytest.approx(1.72, abs=1e-6)
    assert out >= IBKR_TIERED_MIN_ORDER


def test_flat_10bps_ignores_is_ratio_and_leg_price() -> None:
    """`flat_10bps` is notional-based — `is_ratio`/`leg_price` are no-ops."""
    a = commission_per_side(10_000, 20_000.0, model="flat_10bps")
    b = commission_per_side(
        10_000, 20_000.0, model="flat_10bps",
        is_ratio=True, leg_price=100.0,
    )
    assert a == b
    assert a == pytest.approx(20_000.0 * COST_PER_SIDE)


def test_ibkr_lite_ignores_is_ratio_and_leg_price() -> None:
    """`ibkr_lite` is $0 — `is_ratio`/`leg_price` are no-ops."""
    assert commission_per_side(10_000, 20_000.0, model="ibkr_lite") == 0.0
    assert commission_per_side(
        10_000, 20_000.0, model="ibkr_lite",
        is_ratio=True, leg_price=42.0,
    ) == 0.0


def test_single_etf_unchanged_by_is_ratio_false() -> None:
    """Backward-compat: singles use the walker's `shares` directly (and
    ignore `leg_price`, which is irrelevant for singles).
    """
    explicit = commission_per_side(
        200, 10_000.0, model="ibkr_pro_fixed",
        is_ratio=False, leg_price=999.0,
    )
    default = commission_per_side(200, 10_000.0, model="ibkr_pro_fixed")
    # 200 × $0.005 = $1.00 (min binds), cap = $100 — neither binds tightly
    assert explicit == pytest.approx(
        max(IBKR_FIXED_MIN_ORDER, 200 * IBKR_FIXED_PER_SHARE),
    )
    assert explicit == default


def test_ratio_cap_can_still_bind_on_pathological_inputs() -> None:
    """If a degenerate cache passes leg_price = $0.01, leg_shares balloons
    and the 1% cap is the safety net. Verify the cap still binds correctly
    via min(base, notional × max_pct).
    """
    out = commission_per_side(
        shares=40_000, notional=20_000.0,
        model="ibkr_pro_fixed", is_ratio=True, leg_price=0.01,
    )
    cap = 20_000.0 * IBKR_FIXED_MAX_PCT
    assert out == pytest.approx(cap)


# ---------------------------------------------------------------------------
# Integration-level: walker cache `is_ratio` + `leg_price` plumbing
# ---------------------------------------------------------------------------


def _make_minimal_cache(
    is_ratio_flag: bool,
    price_level: float,
    leg_price_value: float | None,
    n_bars: int = 20,
) -> dict[str, dict]:
    """One-asset cache with a single entry at bar 1 → exit at bar 5.

    `price_level` sets the synthetic close (often sub-$1 for ratios).
    `leg_price_value=None` omits the leg_price field (forces fallback).
    """
    dates = pd.bdate_range(start="2024-01-02", periods=n_bars)
    close = np.full(n_bars, price_level, dtype=np.float64)
    pos_raw = np.zeros(n_bars, dtype=np.int8)
    pos_raw[1:5] = 1  # in position bars 1..4, exit triggers at bar 5
    entry: dict = {
        "index": dates,
        "close": close,
        "low": close,
        "atr": np.zeros(n_bars, dtype=np.float64),
        "pos_raw": pos_raw,
        "stop_code": 0,
        "stop_param": 0.0,
        "base_weight": 1.0,
        "is_ratio": is_ratio_flag,
    }
    if leg_price_value is not None:
        entry["leg_price"] = np.full(n_bars, leg_price_value, dtype=np.float64)
    return {"X": entry}


def _full_weight_oracle(active_set: set[str], _date):
    if not active_set:
        return {}
    return {t: 1.0 for t in active_set}


def test_walker_ratio_leg_price_routes_to_real_share_count() -> None:
    """End-to-end: a $0.50 synthetic price with a $50 numerator leg_price
    should produce per-trade commission ~$4 ($2 entry + $2 exit). Without
    the ratio flag, the per-share model sees ~40K synthetic shares and the
    1% cap binds at ~$200/side → ~$380 round-trip.
    """
    seed_nav = 20_000.0  # forces ~$20K notional with weight=1.0
    cache_ratio = _make_minimal_cache(
        is_ratio_flag=True, price_level=0.50, leg_price_value=50.0,
    )
    cache_buggy = _make_minimal_cache(
        is_ratio_flag=False, price_level=0.50, leg_price_value=None,
    )
    state_ratio = walk_portfolio_oracle(
        cache_ratio, _full_weight_oracle, seed_nav=seed_nav,
        buffer=CASH_BUFFER, sizing_rule="cash_fraction",
        commission_model="ibkr_pro_fixed",
    )
    state_buggy = walk_portfolio_oracle(
        cache_buggy, _full_weight_oracle, seed_nav=seed_nav,
        buffer=CASH_BUFFER, sizing_rule="cash_fraction",
        commission_model="ibkr_pro_fixed",
    )
    assert len(state_ratio.trades) == 1
    assert len(state_buggy.trades) == 1
    c_ratio = state_ratio.trades[0].commission_dollars
    c_buggy = state_buggy.trades[0].commission_dollars
    # Ratio-aware: 2 × $2 = $4 round-trip (min $1 wouldn't bind; per-share wins)
    assert c_ratio < 10.0, f"ratio-aware commission too high: ${c_ratio:.2f}"
    # Buggy: cap binds at ~$200/side → ~$380
    assert c_buggy > 100.0, f"expected buggy ~$200/side, got ${c_buggy:.2f}"
    assert c_ratio < c_buggy / 30.0


def test_walker_ratio_uses_avg_etf_price_fallback_when_leg_price_absent() -> None:
    """If a strategy sets `is_ratio=True` but doesn't populate `leg_price`,
    the walker should fall back to AVG_ETF_PRICE — still much lower than
    the synthetic-share bug state, just less accurate than a real
    numerator price.
    """
    seed_nav = 20_000.0
    cache = _make_minimal_cache(
        is_ratio_flag=True, price_level=0.50, leg_price_value=None,
    )
    state = walk_portfolio_oracle(
        cache, _full_weight_oracle, seed_nav=seed_nav,
        buffer=CASH_BUFFER, sizing_rule="cash_fraction",
        commission_model="ibkr_pro_fixed",
    )
    assert len(state.trades) == 1
    # AVG_ETF_PRICE = $50 → 1-leg, ~$2/side, ~$4 round-trip
    assert state.trades[0].commission_dollars < 10.0
