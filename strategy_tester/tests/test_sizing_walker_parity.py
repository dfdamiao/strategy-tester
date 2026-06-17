"""Parity test: ``sizing.size_cash_fraction`` must match the inline
sizing block in ``walker.walk_portfolio_oracle`` byte-for-byte.

Drift guard rationale: the cash-fraction formula is the audit-blessed live
sizing rule (per ``feedback_s5_cash_fraction_sizing.md``, 2026-04-30).
``sizing.py`` exposes it as a standalone for live ``signal_generator.py``;
``walker.py`` uses it inside the no-rebalance simulation. They must agree.

If walker.py changes the formula (slip placement, buffer cap, share rounding),
this test fails and forces a sync.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy_tester.s5_replay.sizing import size_cash_fraction
from strategy_tester.s5_replay.walker import (
    CASH_BUFFER,
    COST_PER_SIDE,
    SLIP_BPS,
)


def _walker_reference(
    weight: float,
    close: float,
    cash: float,
    buffer: float,
    slip: float,
    tx_cost: float,
) -> int:
    """Mirror of walker.walk_portfolio_oracle lines 288, 306-318 for cash_fraction.

    Edit ONLY when walker.py's sizing block changes — keep this byte-faithful
    to the source so the parity assertion is meaningful.
    """
    cash_pool_t = max(0.0, cash * buffer)              # walker line 288
    target_dollars = weight * cash_pool_t              # walker line 307
    sized_dollars = target_dollars                     # walker line 308
    entry_px = close * (1.0 + slip)                    # walker line 314
    cost_per_share = entry_px * (1.0 + tx_cost)        # walker line 315
    if cost_per_share <= 0:
        return 0
    return int(sized_dollars / cost_per_share)         # walker line 316-318


@pytest.mark.unit
@pytest.mark.parametrize("weight,close,cash", [
    (0.5, 400.0, 10000.0),
    (0.1, 50.0, 100000.0),
    (1.0, 12.34, 5000.0),
    (0.05, 1234.56, 50000.0),
    (0.25, 0.99, 1000.0),       # penny stock edge
    (0.5, 400.0, 0.0),          # zero cash → zero shares
])
def test_sizing_matches_walker(weight: float, close: float, cash: float) -> None:
    """For every case, sizing.size_cash_fraction must produce identical shares
    to the walker.py reference re-implementation.
    """
    df = pd.DataFrame({
        "symbol": ["X"], "action": ["enter"],
        "weight": [weight], "entry_px": [close],
    })
    out = size_cash_fraction(
        df, cash=cash, buffer=CASH_BUFFER, slip=SLIP_BPS, tx_cost=COST_PER_SIDE,
    )
    expected = _walker_reference(
        weight, close, cash,
        buffer=CASH_BUFFER, slip=SLIP_BPS, tx_cost=COST_PER_SIDE,
    )
    assert out.iloc[0]["shares"] == expected, (
        f"sizing drift: got {out.iloc[0]['shares']}, walker would size {expected}"
    )


@pytest.mark.unit
def test_non_entry_rows_get_zero_shares() -> None:
    df = pd.DataFrame({
        "symbol": ["A", "B"], "action": ["enter", "exit"],
        "weight": [0.5, 0.5], "entry_px": [100.0, 100.0],
    })
    out = size_cash_fraction(df, cash=10000.0)
    assert out.iloc[0]["shares"] > 0
    assert out.iloc[1]["shares"] == 0


@pytest.mark.unit
def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"symbol": ["X"], "weight": [0.5]})  # no entry_px
    with pytest.raises(ValueError, match="weight.*entry_px"):
        size_cash_fraction(df, cash=10000.0)
