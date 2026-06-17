"""Regime detection methods."""
from __future__ import annotations
import pandas as pd


def sma_regime(
    prices: pd.Series, fast: int = 50, slow: int = 200,
) -> pd.Series:
    """Rule-based regime: 1=bull (fast>slow), 0=bear."""
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    return (sma_fast > sma_slow).astype(int)


def regime_conditional_metrics(
    returns: pd.Series, labels: pd.Series,
) -> dict[int, dict[str, float]]:
    """Sharpe/CAGR/MaxDD per regime label."""
    from strategy_tester.backtest.metrics import (
        annualized_sharpe, geometric_cagr, max_drawdown,
    )
    result = {}
    for label in sorted(labels.unique()):
        mask = labels == label
        r = returns[mask].values
        if len(r) < 20:
            continue
        result[int(label)] = {
            "sharpe": round(annualized_sharpe(r), 4),
            "cagr": round(geometric_cagr(r), 4),
            "max_dd": round(max_drawdown(r), 4),
            "n_days": len(r),
        }
    return result
