"""Extended risk/performance metrics — wraps quantstats + empyrical.

Provides the union metric set from `quantstats.stats`, `empyrical`, and a few
ports from `pyfolio.timeseries`. Every annualised metric respects the
auto-detected `bars_per_year` from `metrics._bars_per_year` so multi-exchange
cohorts (singles/combined ~530 bars/yr) get the right Sharpe scale.

Input contract: every public function takes a `pd.Series` of equity NAV values
(matches `SchemeResult.equity`), converts internally. Benchmark functions take
two aligned equity series.

Source mapping:
    qs.stats   → 77 functions enumerated 2026-05-17. We re-expose ~50 of them
                 with the bars/yr fix and drop redundancies (alias chains).
    empyrical  → stability_of_timeseries, alpha_beta_aligned.
    pyfolio    → gen_drawdown_table, top_drawdowns (ported in drawdown.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import quantstats.stats as qs

from strategy_tester.s5_replay.metrics import _bars_per_year


def _to_returns(eq: pd.Series) -> pd.Series:
    return eq.pct_change().dropna()


def _safe(fn, *args, **kwargs) -> float:
    try:
        v = fn(*args, **kwargs)
        if isinstance(v, (pd.Series, np.ndarray)):
            v = float(v[-1]) if len(v) else float("nan")
        return float(v)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Single-equity-curve extended metrics
# ---------------------------------------------------------------------------


def extended_metrics(eq: pd.Series) -> dict:
    """Return ~50 risk/performance metrics on a single equity curve.

    Adds to `equity_metrics()` (which returns 15 base scalars) — this is the
    qs + empyrical layer that surfaces PSR/Omega/CVaR/Kelly/tail-ratio/etc.
    """
    nan = float("nan")
    out: dict[str, float] = {}
    if len(eq) < 30 or float(eq.iloc[0]) <= 0:
        return out

    rets = _to_returns(eq)
    bpy = _bars_per_year(eq)

    # ---- Returns ----
    out["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1)
    out["geometric_mean"] = _safe(qs.geometric_mean, rets)
    out["avg_return"] = _safe(qs.avg_return, rets)
    out["avg_win"] = _safe(qs.avg_win, rets)
    out["avg_loss"] = _safe(qs.avg_loss, rets)
    out["best_day"] = _safe(qs.best, rets)
    out["worst_day"] = _safe(qs.worst, rets)
    out["rar"] = _safe(qs.rar, rets, periods=bpy)

    # ---- Ratios (returns-only) ----
    out["smart_sharpe"] = _safe(qs.smart_sharpe, rets, periods=bpy)
    out["smart_sortino"] = _safe(qs.smart_sortino, rets, periods=bpy)
    out["adjusted_sortino"] = _safe(qs.adjusted_sortino, rets, periods=bpy)
    out["omega"] = _safe(qs.omega, rets, required_return=0.0, periods=bpy)
    out["gain_to_pain"] = _safe(qs.gain_to_pain_ratio, rets)
    out["risk_return_ratio"] = _safe(qs.risk_return_ratio, rets)
    out["payoff_ratio"] = _safe(qs.payoff_ratio, rets)
    out["win_loss_ratio"] = _safe(qs.win_loss_ratio, rets)
    out["serenity_index"] = _safe(qs.serenity_index, rets)
    out["upi"] = _safe(qs.upi, rets)  # ulcer performance index

    # ---- Probabilistic ----
    out["probabilistic_sharpe"] = _safe(
        qs.probabilistic_sharpe_ratio, rets, periods=bpy,
    )
    out["probabilistic_sortino"] = _safe(
        qs.probabilistic_sortino_ratio, rets, periods=bpy,
    )
    out["probabilistic_adj_sortino"] = _safe(
        qs.probabilistic_adjusted_sortino_ratio, rets, periods=bpy,
    )

    # ---- Tail risk ----
    out["var_95"] = _safe(qs.value_at_risk, rets, confidence=0.95)
    out["var_99"] = _safe(qs.value_at_risk, rets, confidence=0.99)
    out["cvar_95"] = _safe(qs.conditional_value_at_risk, rets, confidence=0.95)
    out["cvar_99"] = _safe(qs.conditional_value_at_risk, rets, confidence=0.99)
    out["tail_ratio"] = _safe(qs.tail_ratio, rets, cutoff=0.95)
    out["outlier_win_ratio"] = _safe(qs.outlier_win_ratio, rets, quantile=0.99)
    out["outlier_loss_ratio"] = _safe(
        qs.outlier_loss_ratio, rets, quantile=0.01,
    )

    # ---- Trade-stat proxies (returns-only, no real trade log) ----
    out["win_rate"] = _safe(qs.win_rate, rets)
    out["consecutive_wins"] = _safe(qs.consecutive_wins, rets)
    out["consecutive_losses"] = _safe(qs.consecutive_losses, rets)
    out["profit_ratio"] = _safe(qs.profit_ratio, rets)
    out["profit_factor"] = _safe(qs.profit_factor, rets)
    out["cpc_index"] = _safe(qs.cpc_index, rets)
    out["common_sense_ratio"] = _safe(qs.common_sense_ratio, rets)
    out["kelly_criterion"] = _safe(qs.kelly_criterion, rets)
    out["risk_of_ruin"] = _safe(qs.risk_of_ruin, rets)

    # ---- Stability (empyrical: R² of log-cumret OLS fit) ----
    out["stability_of_timeseries"] = _stability_of_timeseries(rets)

    # ---- Autocorrelation ----
    out["autocorr_lag1"] = (
        float(rets.autocorr(lag=1)) if len(rets) > 2 else nan
    )

    # ---- Recovery factor ----
    out["recovery_factor"] = _safe(qs.recovery_factor, rets)

    return out


def _stability_of_timeseries(rets: pd.Series) -> float:
    """R² of OLS fit to log-cumulative returns. Higher = smoother growth.

    Port of `empyrical.stability_of_timeseries` so we don't add the dep solely
    for this one function.
    """
    if len(rets) < 3:
        return float("nan")
    cum_log = np.log1p(rets).cumsum()
    if not np.isfinite(cum_log.iloc[-1]):
        return float("nan")
    x = np.arange(len(cum_log), dtype=float)
    y = cum_log.values
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Benchmark-relative extended metrics
# ---------------------------------------------------------------------------


def extended_vs_benchmark(eq: pd.Series, bench_eq: pd.Series | None) -> dict:
    """Treynor + R² + rolling α/β (last value)."""
    out: dict[str, float] = {}
    if bench_eq is None or len(eq) < 30 or len(bench_eq) < 30:
        return out
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    if len(aligned) < 30:
        return out
    rp = aligned["p"].pct_change().dropna()
    rb = aligned["b"].pct_change().dropna()
    rp, rb = rp.align(rb, join="inner")
    bpy = _bars_per_year(aligned["p"])
    out["treynor"] = _safe(qs.treynor_ratio, rp, rb, periods=bpy)
    out["r_squared"] = _safe(qs.r_squared, rp, rb)
    return out


# ---------------------------------------------------------------------------
# Distribution panels
# ---------------------------------------------------------------------------


def monthly_returns_table(eq: pd.Series) -> pd.DataFrame:
    """Year × Month pivot of arithmetic monthly returns (rounded to 4dp).

    Returns DataFrame with months as columns ('Jan'..'Dec', 'YTD'). Empty
    cells when no data. Used by the monthly-heatmap chart.
    """
    if len(eq) < 2:
        return pd.DataFrame()
    monthly = eq.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(
        index="year", columns="month", values="ret", aggfunc="first",
    )
    pivot.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
        for c in pivot.columns
    ]
    annual = eq.resample("YE").last().pct_change().dropna()
    annual.index = annual.index.year
    pivot["YTD"] = annual.reindex(pivot.index)
    return pivot.round(4)


def annual_returns_series(eq: pd.Series) -> pd.Series:
    """Year-end-to-year-end arithmetic returns, indexed by year (int)."""
    if len(eq) < 2:
        return pd.Series(dtype=float)
    yr = eq.resample("YE").last()
    rets = yr.pct_change()
    first_year = eq.index[0].year
    first_yr_end = yr[yr.index.year == first_year]
    if len(first_yr_end) > 0:
        rets.iloc[0] = float(first_yr_end.iloc[0] / eq.iloc[0] - 1)
    rets = rets.dropna()
    rets.index = rets.index.year
    return rets


# ---------------------------------------------------------------------------
# Rolling series (for rolling.py charts)
# ---------------------------------------------------------------------------


def rolling_sharpe(eq: pd.Series, window_days: int = 252) -> pd.Series:
    """Rolling Sharpe over a calendar-bar window."""
    rets = _to_returns(eq)
    if len(rets) < window_days:
        return pd.Series(dtype=float)
    bpy = _bars_per_year(eq)
    mu = rets.rolling(window_days).mean()
    sd = rets.rolling(window_days).std(ddof=1)
    return (mu / sd * np.sqrt(bpy)).dropna()


def rolling_sortino(eq: pd.Series, window_days: int = 252) -> pd.Series:
    rets = _to_returns(eq)
    if len(rets) < window_days:
        return pd.Series(dtype=float)
    bpy = _bars_per_year(eq)

    def _sortino(s: np.ndarray) -> float:
        dn = s[s < 0]
        if len(dn) < 2:
            return np.nan
        d_std = dn.std(ddof=1)
        if d_std < 1e-12:
            return np.nan
        return s.mean() / d_std * np.sqrt(bpy)

    return rets.rolling(window_days).apply(_sortino, raw=True).dropna()


def rolling_volatility(eq: pd.Series, window_days: int = 252) -> pd.Series:
    rets = _to_returns(eq)
    if len(rets) < window_days:
        return pd.Series(dtype=float)
    bpy = _bars_per_year(eq)
    return (rets.rolling(window_days).std(ddof=1) * np.sqrt(bpy)).dropna()


def rolling_beta(
    eq: pd.Series, bench_eq: pd.Series, window_days: int = 252,
) -> pd.Series:
    """Rolling β computed over `window_days` calendar bars."""
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    rp = aligned["p"].pct_change()
    rb = aligned["b"].pct_change()
    df = pd.concat([rp, rb], axis=1).dropna()
    df.columns = pd.Index(["rp", "rb"])
    cov = df["rp"].rolling(window_days).cov(df["rb"])
    var = df["rb"].rolling(window_days).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).dropna()
