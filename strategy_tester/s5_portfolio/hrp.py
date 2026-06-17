"""Hierarchical Risk Parity. LdP AFML Ch.16."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)
from strategy_tester.registry import register_stage


def _empty(method: str = "hrp") -> dict:
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


def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    inv_diag = 1.0 / np.diag(sub_cov)
    inv_diag[~np.isfinite(inv_diag)] = 0.0
    total = float(inv_diag.sum())
    w = inv_diag / total if total > 0 else np.ones(len(indices)) / len(indices)
    return float(w @ sub_cov @ w)


def _hrp_weights(cov: np.ndarray, n: int) -> np.ndarray:
    """Simple recursive bisection HRP."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    stds = np.sqrt(np.diag(cov))
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if stds[i] > 0 and stds[j] > 0:
                corr[i, j] = cov[i, j] / (stds[i] * stds[j])
    np.fill_diagonal(corr, 1.0)

    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, dist.T)

    condensed = squareform(dist, checks=False)
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)
    link = linkage(condensed, method="single")
    order = leaves_list(link).tolist()

    w = np.ones(n)
    items = [order]
    while items:
        cluster = items.pop()
        if len(cluster) <= 1:
            continue
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        left_var = _cluster_var(cov, left)
        right_var = _cluster_var(cov, right)
        denom = left_var + right_var
        alpha = 1.0 - left_var / denom if denom > 0 else 0.5

        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= 1.0 - alpha

        items.append(left)
        items.append(right)

    return w / w.sum()


@register_stage("s5")
def hrp(
    prices: pd.DataFrame,
    s4_result: pd.DataFrame,
    **config,
) -> dict:
    """Hierarchical Risk Parity."""
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
            "portfolio_method": "hrp",
            "n_pairs": 1,
        }

    ret_df = pd.concat(rets_list, axis=1, keys=pair_names).dropna()
    cov = ret_df.cov().values
    n = len(pair_names)

    w_arr = _hrp_weights(cov, n)
    weights = {pair_names[i]: round(float(w_arr[i]), 6) for i in range(n)}

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
        "portfolio_method": "hrp",
        "n_pairs": n,
    }
