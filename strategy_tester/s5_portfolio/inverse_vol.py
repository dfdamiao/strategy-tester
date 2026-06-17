"""Inverse volatility weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "inverse_vol") -> dict:
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
def inverse_vol(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """Weight by 1/vol — simple risk adjustment."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")
    vols: dict[str, float] = {}
    rets_map: dict[str, pd.Series] = {}
    for p in pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            vol = float(r.std() * np.sqrt(252))
            if vol > 0:
                vols[p["pair"]] = vol
                rets_map[p["pair"]] = r

    if not vols:
        return _empty()

    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total = sum(inv_vols.values())
    weights = {k: v / total for k, v in inv_vols.items()}

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
        "portfolio_method": "inverse_vol",
        "n_pairs": len(weights),
    }
