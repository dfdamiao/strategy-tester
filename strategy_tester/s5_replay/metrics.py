"""Metrics: equity_metrics, vs_spy, summary_rows, spy_summary_row.

Extracted verbatim from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` lines
~743-1148 (2026-04-30). Strategy-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from strategy_tester.s5_replay.walker import ANNUALIZE

if TYPE_CHECKING:
    from strategy_tester.s5_replay.runner import SchemeResult


def _bars_per_year(equity: pd.Series) -> float:
    """Auto-detect bars/year from the equity index calendar span.

    Why this exists: 2026-05-11 audit found that signal_sweeps cohorts whose
    index unions multi-exchange tickers (US + EU `.PA/.DE/.MI/.L`) end up with
    ~530 bars/year, not 252. The hardcoded `ANNUALIZE = 252` then understates
    CAGR by ~2.1× and Sharpe by sqrt(252/530) ≈ 0.69. Auto-detecting from the
    series itself preserves correctness for both 252-density (ratios) and
    higher-density (singles + combined) cohorts.

    Falls back to ANNUALIZE if the index isn't datetime-based or the calendar
    span is too short to estimate reliably (< 0.1 yr).
    """
    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 2:
        return float(ANNUALIZE)
    cal_years = (equity.index[-1] - equity.index[0]).days / 365.25
    if cal_years < 0.1:
        return float(ANNUALIZE)
    return len(equity) / cal_years


# ---------------------------------------------------------------------------
# Single-equity-curve metrics
# ---------------------------------------------------------------------------


def equity_metrics(equity: pd.Series) -> dict:
    """Returns a full risk-metrics dict (Sharpe, CAGR, MaxDD, Calmar, Sortino,
    Ulcer, Martin, vol, hit, days_invested, dd_duration, end_value)."""
    nan = float("nan")
    rets = equity.pct_change().dropna()
    n = len(rets)
    base = {
        "sharpe": nan, "cagr": nan, "max_dd": nan, "calmar": nan,
        "sortino": nan, "ulcer": nan, "martin": nan,
        "vol_ann": nan, "hit_rate": nan, "skew": nan, "kurtosis": nan,
        "max_dd_duration_days": 0, "pct_days_invested": nan,
        "end_value": float(equity.iloc[-1]) if len(equity) else nan,
        "n_days": n,
    }
    if n < 10 or equity.iloc[0] <= 0:
        return base

    std = float(rets.std(ddof=1))
    bars_per_yr = _bars_per_year(equity)
    sqrt_a = float(np.sqrt(bars_per_yr))
    sharpe = float(rets.mean() / std * sqrt_a) if std > 1e-12 else nan
    years = n / bars_per_yr
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else nan
    peak = equity.cummax()
    dd_series = (equity / peak - 1)
    max_dd = float(dd_series.min()) if not pd.isna(dd_series.min()) else nan
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-9 else nan
    vol_ann = std * sqrt_a
    hit_rate = float((rets > 0).mean())

    downside = rets[rets < 0]
    d_std = float(downside.std(ddof=1)) if len(downside) > 1 else nan
    sortino = float(rets.mean() / d_std * sqrt_a) if d_std and d_std > 1e-12 else nan

    # Ulcer Index = sqrt(mean(dd_pct^2)) and Martin = CAGR / Ulcer
    dd_pct = dd_series.clip(upper=0).abs() * 100
    ulcer = float(np.sqrt((dd_pct ** 2).mean())) if len(dd_pct) else nan
    martin = float(cagr * 100 / ulcer) if ulcer and ulcer > 1e-9 else nan

    skew = float(rets.skew()) if n > 3 else nan
    kurt = float(rets.kurtosis()) if n > 3 else nan

    # Max-DD duration in calendar days
    in_dd = dd_series < -1e-9
    if in_dd.any():
        groups = (in_dd != in_dd.shift()).cumsum()
        runs = in_dd.groupby(groups).sum()
        max_dd_dur = int(runs.max())
    else:
        max_dd_dur = 0

    pct_invested = float((rets.abs() > 0).mean())

    return {
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd, "calmar": calmar,
        "sortino": sortino, "ulcer": ulcer, "martin": martin,
        "vol_ann": vol_ann, "hit_rate": hit_rate,
        "skew": skew, "kurtosis": kurt,
        "max_dd_duration_days": max_dd_dur,
        "pct_days_invested": pct_invested,
        "end_value": float(equity.iloc[-1]), "n_days": n,
    }


def vs_spy(equity: pd.Series, spy_eq: pd.Series | None) -> dict:
    """Alpha/beta/IR + tracking_error/up-down capture/excess CAGR vs SPY."""
    nan = float("nan")
    base = {
        "alpha_ann": nan, "beta": nan, "ir_vs_spy": nan,
        "tracking_error": nan, "up_capture": nan, "down_capture": nan,
        "excess_cagr": nan, "corr": nan,
    }
    if spy_eq is None:
        return base
    a = pd.concat([equity, spy_eq], axis=1, join="inner").dropna()
    a.columns = pd.Index(["p", "b"])
    rp, rb = a["p"].pct_change().dropna(), a["b"].pct_change().dropna()
    rp, rb = rp.align(rb, join="inner")
    if len(rp) < 30 or rb.var() <= 0:
        return base
    bars_per_yr = _bars_per_year(a["p"])
    beta = float(rp.cov(rb) / rb.var())
    alpha_daily = float(rp.mean() - beta * rb.mean())
    alpha_ann = alpha_daily * bars_per_yr
    diff = rp - rb
    sqrt_a = float(np.sqrt(bars_per_yr))
    diff_std = float(diff.std(ddof=1))
    te = diff_std * sqrt_a if diff_std > 1e-12 else nan
    ir = float(diff.mean() / diff_std * sqrt_a) if diff_std > 1e-12 else nan
    corr = float(rp.corr(rb))
    up = rb > 0
    dn = rb < 0
    up_cap = (
        float(rp[up].mean() / rb[up].mean())
        if up.any() and rb[up].mean() not in (0, np.nan) else nan
    )
    dn_cap = (
        float(rp[dn].mean() / rb[dn].mean())
        if dn.any() and rb[dn].mean() not in (0, np.nan) else nan
    )
    n = len(rp)
    years = n / bars_per_yr
    if years > 0 and len(a) > 1:
        port_cagr = float((a["p"].iloc[-1] / a["p"].iloc[0]) ** (1 / years) - 1)
        spy_cagr = float((a["b"].iloc[-1] / a["b"].iloc[0]) ** (1 / years) - 1)
        excess_cagr = port_cagr - spy_cagr
    else:
        excess_cagr = nan
    return {
        "alpha_ann": alpha_ann, "beta": beta, "ir_vs_spy": ir,
        "tracking_error": te, "up_capture": up_cap, "down_capture": dn_cap,
        "excess_cagr": excess_cagr, "corr": corr,
    }


# ---------------------------------------------------------------------------
# Aggregation across schemes
# ---------------------------------------------------------------------------


def summary_rows(
    results: "list[SchemeResult]", spy_eq: pd.Series | None,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build the canonical 24-row × ~45-col scheme comparison table.

    NB: The ``spy_eq`` and ``oos_start`` parameters are accepted to mirror
    the original signature; they are not currently used because per-scheme
    OOS metrics already incorporate them via ``run_scheme``.
    """
    del spy_eq, oos_start  # silence unused-arg warning; kept for signature parity
    rows = []
    for r in results:
        mo, mf = r.metrics_oos, r.metrics_full
        rows.append({
            "scheme": r.scheme,
            # ---- OOS metrics ----
            "oos_sharpe": mo["sharpe"],
            "oos_cagr": mo["cagr"],
            "oos_max_dd": mo["max_dd"],
            "oos_calmar": mo["calmar"],
            "oos_sortino": mo["sortino"],
            "oos_ulcer": mo["ulcer"],
            "oos_martin": mo["martin"],
            "oos_vol_ann": mo["vol_ann"],
            "oos_hit_rate": mo["hit_rate"],
            "oos_skew": mo["skew"],
            "oos_kurtosis": mo["kurtosis"],
            "oos_dd_dur_days": mo["max_dd_duration_days"],
            "oos_pct_invested": mo["pct_days_invested"],
            "oos_end_value": mo["end_value"],
            # ---- Full-period metrics (parallel to OOS) ----
            "full_sharpe": mf["sharpe"],
            "full_cagr": mf["cagr"],
            "full_max_dd": mf["max_dd"],
            "full_calmar": mf["calmar"],
            "full_sortino": mf["sortino"],
            "full_ulcer": mf["ulcer"],
            "full_martin": mf["martin"],
            "full_vol_ann": mf["vol_ann"],
            "full_hit_rate": mf["hit_rate"],
            "full_skew": mf["skew"],
            "full_kurtosis": mf["kurtosis"],
            "full_dd_dur_days": mf["max_dd_duration_days"],
            "full_pct_invested": mf["pct_days_invested"],
            "end_value": mf["end_value"],
            # ---- vs SPY (OOS window) ----
            "ir_vs_spy": r.vs_spy_oos["ir_vs_spy"],
            "alpha_ann": r.vs_spy_oos["alpha_ann"],
            "beta": r.vs_spy_oos["beta"],
            "corr_spy": r.vs_spy_oos["corr"],
            "tracking_error": r.vs_spy_oos["tracking_error"],
            "up_capture": r.vs_spy_oos["up_capture"],
            "down_capture": r.vs_spy_oos["down_capture"],
            "excess_cagr": r.vs_spy_oos["excess_cagr"],
            # ---- Execution ----
            "n_trades": r.n_trades,
            "n_failed_entries": r.n_failed,
            "n_clipped": r.n_clipped,
            "n_too_small": r.n_too_small,
            "mean_clip_ratio": r.mean_clip_ratio,
            # Cap-verification diagnostics. For paleologo_strict /
            # cash_fraction_*_capped these should be ≤ cap × 100 × buffer.
            "max_entry_pct_cash": r.max_entry_pct_cash,
            "p95_entry_pct_cash": r.p95_entry_pct_cash,
            "max_entry_pct_nav": r.max_entry_pct_nav,
            # Concentration + cap-binding telemetry (2026-05-18, ref-doc
            # Q5/Q17/Q21). HHI = Σw²; ENB = 1/HHI (Meucci 2009 lower-bound).
            # cap_bound_rate = how often cap actually bound at entry.
            "mean_hhi": r.mean_hhi,
            "max_hhi": r.max_hhi,
            "mean_enb": r.mean_enb,
            "min_enb": r.min_enb,
            "n_cap_bound": r.n_cap_bound,
            "n_entries_total": r.n_entries_total,
            "cap_bound_rate": r.cap_bound_rate,
        })
    return pd.DataFrame(rows)


def spy_summary_row(
    spy_eq: pd.Series | None, oos_start: pd.Timestamp, columns: list[str],
    seed_nav: float | None = None,
) -> dict | None:
    """Build a SPY-as-benchmark row for the scheme comparison table.

    Computes SPY's own OOS + full-period metrics so it can be displayed
    alongside the schemes for direct visual comparison. Returns None if
    spy_eq is unavailable.

    seed_nav (added 2026-05-10 cum_rsi_v2): when provided, rebase the OOS
    slice so ``oos_end_value`` reflects OOS-only growth from seed_nav rather
    than the absolute final value of the continuous spy_eq curve. Without
    rebasing, ``spy_oos.iloc[-1] == spy_eq.iloc[-1]`` for any slice that
    extends to today, making oos_end_value indistinguishable from
    full_end_value. Sharpe/CAGR/MaxDD are scale-invariant so they were
    already correct; only end_value needed the fix.
    """
    if spy_eq is None or len(spy_eq) == 0:
        return None
    spy_oos = spy_eq.loc[spy_eq.index >= oos_start]
    if len(spy_oos) == 0:
        return None
    if seed_nav is not None and float(spy_oos.iloc[0]) > 0:
        spy_oos_for_metrics = spy_oos / float(spy_oos.iloc[0]) * seed_nav
    else:
        spy_oos_for_metrics = spy_oos
    mo = equity_metrics(spy_oos_for_metrics)
    mf = equity_metrics(spy_eq)
    nan = float("nan")
    row: dict = {col: nan for col in columns}
    row["scheme"] = "SPY"
    # OOS
    row["oos_sharpe"] = mo["sharpe"]
    row["oos_cagr"] = mo["cagr"]
    row["oos_max_dd"] = mo["max_dd"]
    row["oos_calmar"] = mo["calmar"]
    row["oos_sortino"] = mo["sortino"]
    row["oos_ulcer"] = mo["ulcer"]
    row["oos_martin"] = mo["martin"]
    row["oos_vol_ann"] = mo["vol_ann"]
    row["oos_hit_rate"] = mo["hit_rate"]
    row["oos_skew"] = mo["skew"]
    row["oos_kurtosis"] = mo["kurtosis"]
    row["oos_dd_dur_days"] = mo["max_dd_duration_days"]
    row["oos_pct_invested"] = mo["pct_days_invested"]
    row["oos_end_value"] = mo["end_value"]
    # Full
    row["full_sharpe"] = mf["sharpe"]
    row["full_cagr"] = mf["cagr"]
    row["full_max_dd"] = mf["max_dd"]
    row["full_calmar"] = mf["calmar"]
    row["full_sortino"] = mf["sortino"]
    row["full_ulcer"] = mf["ulcer"]
    row["full_martin"] = mf["martin"]
    row["full_vol_ann"] = mf["vol_ann"]
    row["full_hit_rate"] = mf["hit_rate"]
    row["full_skew"] = mf["skew"]
    row["full_kurtosis"] = mf["kurtosis"]
    row["full_dd_dur_days"] = mf["max_dd_duration_days"]
    row["full_pct_invested"] = mf["pct_days_invested"]
    row["end_value"] = mf["end_value"]
    # SPY vs itself: trivial defaults so columns don't show NaN
    row["ir_vs_spy"] = 0.0
    row["alpha_ann"] = 0.0
    row["beta"] = 1.0
    row["corr_spy"] = 1.0
    row["tracking_error"] = 0.0
    row["up_capture"] = 1.0
    row["down_capture"] = 1.0
    row["excess_cagr"] = 0.0
    # Execution columns stay NaN (SPY isn't traded by the strategy)
    return row
