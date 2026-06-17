"""Half-Kelly position sizing. Kelly (1956), Thorp (2006), Chan AT (2013) Ch.6."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "half_kelly") -> dict:
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
def half_kelly(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """f = mu / (2 * vol^2), scaled to sum=1. Chan AT (2013) Ch.6."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")
    kellys: dict[str, float] = {}
    rets_map: dict[str, pd.Series] = {}

    for p in pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            mu = float(r.mean() * 252)
            vol = float(r.std() * np.sqrt(252))
            if vol > 0:
                f = mu / (2.0 * vol**2)
                kellys[p["pair"]] = max(0.0, f)
                rets_map[p["pair"]] = r

    total = sum(kellys.values())
    if total <= 0:
        n = len(rets_map)
        if n == 0:
            return _empty()
        weights = {k: 1.0 / n for k in rets_map}
    else:
        weights = {k: v / total for k, v in kellys.items() if k in rets_map}

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
        "portfolio_method": "half_kelly",
        "n_pairs": len(weights),
    }
