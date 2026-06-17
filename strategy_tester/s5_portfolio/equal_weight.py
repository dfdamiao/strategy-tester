"""Equal weight portfolio. DeMiguel (2009)."""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty_result(method: str) -> dict:
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
def equal_weight(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """1/N weight for each passing pair."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty_result("equal_weight")

    pairs = passed.to_dict("records")
    n = len(pairs)
    weights = {p["pair"]: 1.0 / n for p in pairs}

    all_rets = []
    for p in pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            all_rets.append(r * weights[p["pair"]])

    if not all_rets:
        return _empty_result("equal_weight")

    combined = pd.concat(all_rets, axis=1).sum(axis=1).dropna()
    equity = (1 + combined).cumprod()

    return {
        "equity_curve": equity,
        "daily_returns": combined,
        "sharpe": round(annualized_sharpe(combined.values), 4),
        "cagr": round(geometric_cagr(combined.values), 4),
        "max_dd": round(max_drawdown(combined.values), 4),
        "weights": weights,
        "portfolio_method": "equal_weight",
        "n_pairs": n,
    }
