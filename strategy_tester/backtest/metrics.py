"""Performance metrics for backtesting.

Functions compute annualized metrics from daily return arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def annualized_sharpe(
    returns: np.ndarray, risk_free: float = 0.0,
) -> float:
    """mean(r-rf) / std(r, ddof=1) * sqrt(252). Full period incl zeros."""
    r = np.asarray(returns, dtype=np.float64)
    excess = r - risk_free / 252
    std = np.std(excess, ddof=1)
    if std == 0 or len(r) == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def geometric_cagr(returns: np.ndarray) -> float:
    """prod(1+r)^(252/n) - 1."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return 0.0
    cum = np.prod(1 + r)
    if cum <= 0:
        return -1.0
    return float(cum ** (252 / n) - 1)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough decline (negative number)."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return float(np.min(dd))


def calmar_ratio(returns: np.ndarray) -> float:
    """CAGR / |MaxDD|."""
    cagr = geometric_cagr(returns)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(cagr / abs(mdd))


def sortino_ratio(returns: np.ndarray) -> float:
    """CAGR / downside_vol (annualized)."""
    r = np.asarray(returns, dtype=np.float64)
    cagr = geometric_cagr(r)
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    downside_vol = float(np.std(neg, ddof=1) * np.sqrt(252))
    if downside_vol == 0:
        return 0.0
    return float(cagr / downside_vol)


def psr_stat(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Bailey & LdP (2012) Probabilistic Sharpe Ratio.
    Returns probability that true SR > sr_benchmark.
    PSR = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2))
    """
    if n_obs < 2:
        return 0.0
    denom_sq = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    if denom_sq <= 0:
        return 0.0
    z = (sharpe - sr_benchmark) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(denom_sq)
    return float(norm.cdf(z))
