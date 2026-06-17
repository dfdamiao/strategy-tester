"""Benchmark comparison metrics."""
from __future__ import annotations
import numpy as np
from scipy.stats import linregress


def information_ratio(
    returns: np.ndarray, benchmark_returns: np.ndarray,
) -> float:
    """IR = mean(excess) / std(excess)."""
    excess = np.asarray(returns) - np.asarray(benchmark_returns)
    std = np.std(excess, ddof=1)
    if std == 0 or len(excess) == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def jensen_alpha(
    returns: np.ndarray, benchmark_returns: np.ndarray,
) -> float:
    """CAPM alpha (annualized)."""
    r = np.asarray(returns)
    b = np.asarray(benchmark_returns)
    n = min(len(r), len(b))
    if n < 30:
        return 0.0
    result = linregress(b[:n], r[:n])
    return float(result.intercept * 252)


def treynor_ratio(
    returns: np.ndarray, benchmark_returns: np.ndarray,
    risk_free: float = 0.0,
) -> float:
    """Excess return / beta."""
    r = np.asarray(returns)
    b = np.asarray(benchmark_returns)
    n = min(len(r), len(b))
    if n < 30:
        return 0.0
    result = linregress(b[:n], r[:n])
    beta = result.slope
    if abs(beta) < 1e-9:
        return 0.0
    excess = np.mean(r[:n]) - risk_free / 252
    return float(excess / beta * 252)


def up_down_capture(
    returns: np.ndarray, benchmark_returns: np.ndarray,
) -> tuple[float, float]:
    """Up-capture and down-capture ratios."""
    r = np.asarray(returns)
    b = np.asarray(benchmark_returns)
    n = min(len(r), len(b))
    r, b = r[:n], b[:n]

    up_mask = b > 0
    down_mask = b < 0

    up_cap = (
        float(np.mean(r[up_mask]) / np.mean(b[up_mask]))
        if up_mask.any() and np.mean(b[up_mask]) != 0
        else 0.0
    )
    down_cap = (
        float(np.mean(r[down_mask]) / np.mean(b[down_mask]))
        if down_mask.any() and np.mean(b[down_mask]) != 0
        else 0.0
    )
    return up_cap, down_cap


def fundamental_law(
    ic: float, breadth: float, tc: float = 1.0,
) -> float:
    """Grinold Fundamental Law: IR = IC * sqrt(BR) * TC."""
    return ic * np.sqrt(breadth) * tc
