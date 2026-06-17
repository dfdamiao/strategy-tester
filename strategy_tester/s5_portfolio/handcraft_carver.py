"""Handcraft weighting. Carver Systematic Trading Ch.4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "handcraft_carver") -> dict:
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


def _inv_vol_within_group(
    group_pairs: list[dict], prices: pd.DataFrame
) -> dict[str, float]:
    """Inverse vol weights within a group, renormalized."""
    vols: dict[str, float] = {}
    for p in group_pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            vol = float(r.std() * np.sqrt(252))
            if vol > 0:
                vols[p["pair"]] = vol
    if not vols:
        return {}
    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total = sum(inv_vols.values())
    return {k: v / total for k, v in inv_vols.items()}


@register_stage("s5")
def handcraft_carver(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """Group by sector if available; inv-vol within group; equal across groups."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")

    # Group by sector column if present
    has_sector = "sector" in passed.columns
    if has_sector:
        sectors: dict[str, list[dict]] = {}
        for p in pairs:
            sector = p.get("sector", "default")
            sectors.setdefault(sector, []).append(p)
    else:
        sectors = {"default": pairs}

    n_groups = len(sectors)
    group_weight = 1.0 / n_groups

    final_weights: dict[str, float] = {}
    rets_map: dict[str, pd.Series] = {}

    for sector, group_pairs in sectors.items():
        within = _inv_vol_within_group(group_pairs, prices)
        for pair_key, within_w in within.items():
            final_weights[pair_key] = within_w * group_weight

        for p in group_pairs:
            num = p["numerator"]
            if num in prices.columns and p["pair"] in final_weights:
                rets_map[p["pair"]] = prices[num].pct_change().dropna()

    if not final_weights:
        return _empty()

    # Renormalize
    total = sum(final_weights.values())
    weights = {k: v / total for k, v in final_weights.items()}

    all_rets = [rets_map[k] * weights[k] for k in weights if k in rets_map]
    combined = pd.concat(all_rets, axis=1).sum(axis=1).dropna()
    equity = (1 + combined).cumprod()

    return {
        "equity_curve": equity,
        "daily_returns": combined,
        "sharpe": round(annualized_sharpe(combined.values), 4),
        "cagr": round(geometric_cagr(combined.values), 4),
        "max_dd": round(max_drawdown(combined.values), 4),
        "weights": weights,
        "portfolio_method": "handcraft_carver",
        "n_pairs": len(weights),
    }
