"""Sharpe-weighted portfolio."""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "sharpe_weighted") -> dict:
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
def sharpe_weighted(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """Weight = max(0, SR_i) / sum(max(0, SR_j))."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")
    sharpes: dict[str, float] = {}
    rets_map: dict[str, pd.Series] = {}

    use_oos = "oos_sharpe" in s4_result.columns
    for p in pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            if use_oos:
                # Prefer OOS Sharpe from S4 (no lookahead)
                sr = float(p.get("oos_sharpe", 0.0))
            else:
                # WARNING: lookahead on full period
                sr = annualized_sharpe(r.values)
            sharpes[p["pair"]] = max(0.0, sr)
            rets_map[p["pair"]] = r

    total = sum(sharpes.values())
    if total <= 0:
        # Fall back to equal weight if all SRs non-positive
        n = len(rets_map)
        if n == 0:
            return _empty()
        weights = {k: 1.0 / n for k in rets_map}
    else:
        weights = {k: v / total for k, v in sharpes.items() if k in rets_map}

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
        "portfolio_method": "sharpe_weighted",
        "n_pairs": len(weights),
    }
