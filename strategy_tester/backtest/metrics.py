"""Performance metrics for backtesting.

Functions compute annualized metrics from daily return arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def annualized_sharpe(
    returns: np.ndarray,
    risk_free: float = 0.0,
    ddof: int = 1,
    periods_per_year: float = 252.0,
) -> float:
    """CANONICAL Sharpe kernel (2026-08-03 dedup): mean(r-rf) / std(r, ddof)
    * sqrt(periods_per_year). Full period incl zeros.

    ddof=1 (sample std — Lo 2002 / Bailey-LdP convention) is the house
    standard; hand-rolled numpy ``.std()`` (ddof=0) is a systematic
    sqrt(n/(n-1)) inflation at the DSR-95 gate. ``periods_per_year``
    defaults to 252; pass ``bars_per_year_from_index(index)`` for
    multi-exchange cohorts (~530 bars/yr).
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    std = np.std(excess, ddof=ddof)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def bars_per_year_from_index(index, fallback: float = 252.0) -> float:
    """Auto-detect bars/year from a (Datetime)Index calendar span.

    Frequency-correct annualization primitive (2026-08-05): the engine's
    Sharpe is 252-annualized regardless of bar frequency — detect the true
    bars/year from the panel instead. Falls back on non-datetime indexes
    or spans < 0.1 yr.
    """
    import pandas as pd

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return float(fallback)
    cal_years = (index[-1] - index[0]).days / 365.25
    if cal_years < 0.1:
        return float(fallback)
    return float(len(index) / cal_years)


def geometric_cagr(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """prod(1+r)^(periods_per_year/n) - 1 (252 = daily-bar default)."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return 0.0
    cum = np.prod(1 + r)
    if cum <= 0:
        return -1.0
    return float(cum ** (periods_per_year / n) - 1)


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
    denom_sq = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if denom_sq <= 0:
        return 0.0
    z = (sharpe - sr_benchmark) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


# Default annualization for daily-bar strategies. The significance path
# (WFA/CPCV mean_test_sharpe) is always annualized with this factor before
# it reaches PSR/DSR, so this is the factor to undo.
PERIODS_PER_YEAR: int = 252


def to_per_period(
    sharpe_ann: float,
    sr_variance_ann: float | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[float, float | None]:
    """De-annualize an annualized Sharpe (and its cross-trial variance) so it
    is frequency-consistent with a per-BAR ``n_obs``.

    Bailey-LdP PSR/DSR require ``SR_hat`` expressed at the SAME frequency as
    the ``n`` observations in ``sqrt(n-1)``. The significance path annualizes
    SR (``mean/std*sqrt(252)``) yet counts ``n_obs`` in daily bars. Pairing
    annualized SR with a daily bar count inflates the z-statistic by
    ~sqrt(252). Callers that hold a daily bar count MUST de-annualize the SR
    (and the variance-of-SR, which scales by ``1/ppy``) via this helper
    before ``psr_stat`` / ``deflated_sharpe``.

    Returns ``(sharpe_pp, sr_variance_pp)`` where ``sr_variance_pp`` is
    ``None`` if no variance was supplied.
    """
    sr_pp = float(sharpe_ann) / np.sqrt(periods_per_year)
    var_pp = (
        None if sr_variance_ann is None else float(sr_variance_ann) / periods_per_year
    )
    return sr_pp, var_pp
