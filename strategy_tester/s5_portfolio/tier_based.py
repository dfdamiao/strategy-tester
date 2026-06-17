"""Tier-based weighting (TOP=2x, SECOND=1x, REJECT-excluded)."""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage

_TIER_MULTIPLIER = {
    "TOP_TIER": 2.0,
    "SECOND_TIER": 1.0,
    "REJECT": 0.0,
}


def _empty(method: str = "tier_based") -> dict:
    return {
        "equity_curve": pd.Series(dtype=float),
        "daily_returns": pd.Series(dtype=float),
        "sharpe": 0.0,
        "cagr": 0.0,
        "max_dd": 0.0,
        "weights": {},
        "portfolio_method": method,
        "n_pairs": 0,
    }


@register_stage("s5")
def tier_based(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """TOP=2x base weight, SECOND=1x, unrecognized tiers=1x, normalized."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")
    raw_weights: dict[str, float] = {}
    rets_map: dict[str, pd.Series] = {}

    for p in pairs:
        tier = p.get("tier", "SECOND_TIER")
        mult = _TIER_MULTIPLIER.get(tier, 1.0)
        if mult <= 0:
            continue
        num = p["numerator"]
        if num in prices.columns:
            raw_weights[p["pair"]] = mult
            rets_map[p["pair"]] = prices[num].pct_change().dropna()

    if not raw_weights:
        return _empty()

    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}

    all_rets = [rets_map[k] * weights[k] for k in weights]
    combined = pd.concat(all_rets, axis=1).sum(axis=1).dropna()
    equity = (1 + combined).cumprod()

    return {
        "equity_curve": equity,
        "daily_returns": combined,
        "sharpe": round(annualized_sharpe(combined.values), 4),
        "cagr": round(geometric_cagr(combined.values), 4),
        "max_dd": round(max_drawdown(combined.values), 4),
        "weights": weights,
        "portfolio_method": "tier_based",
        "n_pairs": len(weights),
    }
