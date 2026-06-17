"""Equal Risk Contribution (risk parity)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "risk_parity") -> dict:
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
def risk_parity(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """Equal risk contribution via iterative solver."""
    passed = s4_result[s4_result["passed"]]
    if passed.empty:
        return _empty()

    pairs = passed.to_dict("records")
    pair_names: list[str] = []
    rets_list: list[pd.Series] = []
    for p in pairs:
        num = p["numerator"]
        if num in prices.columns:
            r = prices[num].pct_change().dropna()
            pair_names.append(p["pair"])
            rets_list.append(r)

    if not pair_names:
        return _empty()

    if len(pair_names) < 2:
        weights = {pair_names[0]: 1.0}
        combined = rets_list[0]
        equity = (1 + combined).cumprod()
        return {
            "equity_curve": equity,
            "daily_returns": combined,
            "sharpe": round(annualized_sharpe(combined.values), 4),
            "cagr": round(geometric_cagr(combined.values), 4),
            "max_dd": round(max_drawdown(combined.values), 4),
            "weights": weights,
            "portfolio_method": "risk_parity",
            "n_pairs": 1,
        }

    ret_df = pd.concat(rets_list, axis=1, keys=pair_names).dropna()
    cov = ret_df.cov().values
    n = len(pair_names)

    w = np.ones(n) / n
    for _ in range(100):
        risk_contrib = w * (cov @ w)
        total_risk = float(w @ cov @ w)
        if total_risk <= 0:
            break
        target = total_risk / n
        for i in range(n):
            if risk_contrib[i] > 0:
                w[i] *= target / risk_contrib[i]
        w = w / w.sum()

    weights = {pair_names[i]: round(float(w[i]), 6) for i in range(n)}
    all_rets = [ret_df[k] * weights[k] for k in pair_names]
    combined = pd.concat(all_rets, axis=1).sum(axis=1)
    equity = (1 + combined).cumprod()

    return {
        "equity_curve": equity,
        "daily_returns": combined,
        "sharpe": round(annualized_sharpe(combined.values), 4),
        "cagr": round(geometric_cagr(combined.values), 4),
        "max_dd": round(max_drawdown(combined.values), 4),
        "weights": weights,
        "portfolio_method": "risk_parity",
        "n_pairs": n,
    }
