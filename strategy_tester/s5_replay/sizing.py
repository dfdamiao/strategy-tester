"""Live cash-fraction position sizing — audit-blessed standalone.

Mirrors the inline sizing block in ``walker.walk_portfolio_oracle`` (lines
288, 306-318) so live ``signal_generator.py`` calls a single function instead
of every strategy carrying its own copy of the formula.

Audit context (``feedback_s5_cash_fraction_sizing.md``, 2026-04-30): silent
``target = w × NetLiq`` clipped sizing dropped OBV-pivot's realised Sharpe
from 1.557 to 0.912 (-41%). The rule is now mandatory: ``target = w × cash
× buffer``, snapshot ``cash`` BEFORE the entry-sort loop so ordering can't
re-introduce first-mover cash-depletion bias.

Drift guard: ``tests/test_sizing_walker_parity.py`` pins this implementation
to byte-identical output vs ``walker.walk_portfolio_oracle`` on a small
fixture. If walker.py changes the formula, the test fails — never let the
two diverge silently.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.s5_replay.walker import (
    CASH_BUFFER,
    COST_PER_SIDE,
    SLIP_BPS,
)


def size_cash_fraction(
    actions: pd.DataFrame,
    *,
    cash: float,
    buffer: float = CASH_BUFFER,
    slip: float = SLIP_BPS,
    tx_cost: float = COST_PER_SIDE,
) -> pd.DataFrame:
    """Size pending entries via the audit-blessed ``cash_fraction`` rule.

    Parameters
    ----------
    actions : DataFrame
        Must contain columns ``weight`` (float, 0..1) and ``entry_px`` (float).
        ``entry_px`` is the **pre-slippage reference price** (close at
        as-of bar). Slippage and tx_cost are applied internally to match
        ``walker.walk_portfolio_oracle`` exactly. Other columns
        (symbol/pair/side/action) are passed through untouched. Action rows
        where ``action != "enter"`` are passed through with NaN sizing
        fields — they don't consume cash.
    cash : float
        Current cash balance, snapshot BEFORE the loop. Live executor must
        query this once and pass the snapshot — do not requery per entry.
    buffer : float, default ``CASH_BUFFER`` (0.95)
        Cash bucket multiplier. ``cash_pool = cash × buffer``.
    slip : float, default ``SLIP_BPS`` (5 bps)
        Slippage applied to entry_px before share calc.
    tx_cost : float, default ``COST_PER_SIDE`` (10 bps)
        Per-side transaction cost.

    Returns
    -------
    DataFrame
        Same rows as ``actions``, with three new columns appended:
            target_dollars : w × cash_pool
            cost_per_share : entry_px × (1 + slip) × (1 + tx_cost)
            shares         : int(target_dollars / cost_per_share), 0 if no cash
    """
    if "weight" not in actions.columns or "entry_px" not in actions.columns:
        raise ValueError("actions must have columns 'weight' and 'entry_px'")

    out = actions.copy()
    cash_pool = max(0.0, cash * buffer)
    if "action" in out.columns:
        is_entry = pd.Series(out["action"]).eq("enter")
    else:
        is_entry = pd.Series([True] * len(out), index=out.index)

    target_dollars = (out["weight"] * cash_pool).where(is_entry, other=float("nan"))
    cost_per_share = (out["entry_px"] * (1.0 + slip) * (1.0 + tx_cost)).where(
        is_entry, other=float("nan")
    )
    shares = (target_dollars / cost_per_share).where(cost_per_share > 0, other=0.0)
    shares = shares.fillna(0.0).astype(int).where(is_entry, other=0)

    out["target_dollars"] = target_dollars
    out["cost_per_share"] = cost_per_share
    out["shares"] = shares
    return out
