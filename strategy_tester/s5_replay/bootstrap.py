"""Bootstrap perf-stats CI + non-parametric forecast cone.

Port of `pyfolio.timeseries.calc_bootstrap` and `forecast_cone_bootstrap`,
adapted to take an equity series (not returns) and honour our auto-detected
bars/year. Resamples by daily-return draw with replacement.

Numbers calibrated to match pyfolio defaults: n_samples=1000 for perf-stats
CI; 1000 path simulations × 252 forward bars for the cone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.s5_replay.metrics import _bars_per_year

_RNG = np.random.default_rng(0)  # deterministic for replay reports


def _bootstrap_returns(rets: np.ndarray, n_samples: int) -> np.ndarray:
    """Return shape (n_samples, len(rets)) of resampled returns."""
    idx = _RNG.integers(0, len(rets), size=(n_samples, len(rets)))
    return rets[idx]


def perf_stats_bootstrap(
    eq: pd.Series, n_samples: int = 1000,
) -> pd.DataFrame:
    """Bootstrap distribution of (Sharpe, Sortino, Calmar, CAGR, MaxDD,
    Omega, VolAnn) by daily-return resampling.

    Returns DataFrame indexed by sample id, columns = metrics. Use
    `.describe()` or per-column .quantile() for CIs.
    """
    if len(eq) < 60:
        return pd.DataFrame()
    rets = eq.pct_change().dropna().values
    if len(rets) < 60:
        return pd.DataFrame()
    bpy = _bars_per_year(eq)
    sqrt_a = np.sqrt(bpy)
    samples = _bootstrap_returns(rets, n_samples)

    sharpe = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * sqrt_a
    vol = samples.std(axis=1, ddof=1) * sqrt_a

    # Downside-only std for Sortino
    dn = np.where(samples < 0, samples, np.nan)
    d_std = np.nanstd(dn, axis=1, ddof=1)
    sortino = np.where(
        d_std > 1e-12, samples.mean(axis=1) / d_std * sqrt_a, np.nan,
    )

    # Per-sample equity curves for CAGR / MaxDD / Calmar
    cum = (1 + samples).cumprod(axis=1)
    years = len(rets) / bpy
    cagr = cum[:, -1] ** (1 / years) - 1 if years > 0 else np.full(n_samples, np.nan)
    peak = np.maximum.accumulate(cum, axis=1)
    dd = cum / peak - 1
    max_dd = dd.min(axis=1)
    calmar = np.where(max_dd < -1e-9, cagr / np.abs(max_dd), np.nan)

    # Omega at threshold 0
    gains = np.where(samples > 0, samples, 0).sum(axis=1)
    losses = np.where(samples < 0, -samples, 0).sum(axis=1)
    omega = np.where(losses > 1e-12, gains / losses, np.nan)

    return pd.DataFrame({
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "max_dd": max_dd,
        "vol_ann": vol,
        "omega": omega,
    })


def perf_stats_ci(
    eq: pd.Series, n_samples: int = 1000,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Per-metric quantile table from `perf_stats_bootstrap`."""
    samples = perf_stats_bootstrap(eq, n_samples=n_samples)
    if samples.empty:
        return pd.DataFrame()
    return samples.quantile(list(quantiles)).T


def forecast_cone_bootstrap(
    eq: pd.Series,
    forward_days: int = 252,
    n_samples: int = 1000,
    starting_value: float | None = None,
    cone_std: tuple[float, ...] = (1.0, 2.0),
) -> pd.DataFrame:
    """Non-parametric forecast cone projected from `eq.index[-1]`.

    Resample empirical daily returns with replacement, simulate `forward_days`
    bars × `n_samples` paths, then compute quantiles at the cone_std levels
    interpreted as normal-equivalent (1σ ≈ 68%, 2σ ≈ 95%).

    Returns DataFrame indexed by forward business-day date with columns:
        median, lower_1sd, upper_1sd, lower_2sd, upper_2sd
    """
    if len(eq) < 60:
        return pd.DataFrame()
    rets = eq.pct_change().dropna().values
    if len(rets) < 60:
        return pd.DataFrame()
    start_val = float(eq.iloc[-1]) if starting_value is None else float(starting_value)

    samples = _bootstrap_returns(rets, n_samples)[:, :forward_days]
    paths = start_val * (1 + samples).cumprod(axis=1)

    qs_lo_1 = np.quantile(paths, 0.16, axis=0)
    qs_hi_1 = np.quantile(paths, 0.84, axis=0)
    qs_lo_2 = np.quantile(paths, 0.025, axis=0)
    qs_hi_2 = np.quantile(paths, 0.975, axis=0)
    qs_med = np.quantile(paths, 0.5, axis=0)

    last_date = eq.index[-1]
    fwd_index = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=forward_days,
    )
    return pd.DataFrame(
        {
            "median": qs_med,
            "lower_1sd": qs_lo_1,
            "upper_1sd": qs_hi_1,
            "lower_2sd": qs_lo_2,
            "upper_2sd": qs_hi_2,
        },
        index=fwd_index,
    )
