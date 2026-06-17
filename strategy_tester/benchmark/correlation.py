"""Correlation analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd


def static_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix."""
    return returns_df.corr()


def rolling_correlation(
    returns_df: pd.DataFrame, window: int = 63,
) -> pd.DataFrame:
    """Rolling pairwise correlation (mean across pairs)."""
    return returns_df.rolling(window).corr().groupby(level=0).mean()


def tail_correlation(
    returns_df: pd.DataFrame, threshold: float = 0.05,
) -> pd.DataFrame:
    """Correlation in tails only (worst/best days)."""
    lower = returns_df.quantile(threshold)
    upper = returns_df.quantile(1 - threshold)
    tail_mask = (returns_df <= lower) | (returns_df >= upper)
    tail_rets = returns_df[tail_mask.any(axis=1)]
    if len(tail_rets) < 10:
        return returns_df.corr()  # fallback
    return tail_rets.corr()


def diversification_ratio(
    weights: np.ndarray, cov_matrix: np.ndarray,
) -> float:
    """Choueifaty DR = weighted avg vol / portfolio vol."""
    w = np.asarray(weights)
    cov = np.asarray(cov_matrix)
    stds = np.sqrt(np.diag(cov))
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol == 0:
        return 0.0
    return float((w @ stds) / port_vol)


def diversification_multiplier(
    weights: np.ndarray, corr_matrix: np.ndarray,
) -> float:
    """Carver DM = 1 / sqrt(w' C w)."""
    w = np.asarray(weights)
    c = np.asarray(corr_matrix)
    denom = w @ c @ w
    if denom <= 0:
        return 1.0
    return float(1.0 / np.sqrt(denom))
