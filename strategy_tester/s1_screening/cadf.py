"""Cointegrating ADF (CADF) test. Chan, Algorithmic Trading (2013) Ch.2.

Unlike standard ADF on the ratio, CADF first estimates the hedge ratio
via OLS regression (A = beta * B + residual), then runs ADF on the
residual. This tests whether the pair is cointegrated — i.e., there
exists a linear combination that is stationary.

The hedge ratio beta allows for non-1:1 relationships between A and B,
making this more general than testing the simple ratio A/B.

Reference:
    Engle & Granger (1987), "Co-integration and error correction:
    Representation, estimation, and testing", Econometrica 55(2).
    Chan, Algorithmic Trading (2013) Ch.2 — practical implementation.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numba
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage
from strategy_tester.s2_signal.kalman_hedge import _kalman_beta_filter

# Module-level globals for mp.Pool fork inheritance (avoids pickling DataFrame).
# Set by `cadf_rolling()` before pool fork; inherited via COW in workers.
_PRICES_FORK: pd.DataFrame | None = None
_CONFIG_FORK: dict | None = None


def _worker_init_fork() -> None:
    """Fork-pool worker initializer: silence warnings, warm numba JIT.

    Globals (_PRICES_FORK, _CONFIG_FORK) are inherited via COW from parent.
    """
    import warnings
    warnings.filterwarnings("ignore")
    _adf_tstat_maxlag1(np.zeros(50, dtype=np.float64))


def _worker_process_pair(pair: dict) -> dict | None:
    """Fork-pool worker: process one pair using inherited globals."""
    cfg = _CONFIG_FORK
    assert cfg is not None and _PRICES_FORK is not None, "globals not set"
    return _process_one_rolling(
        _PRICES_FORK, pair,
        cfg["window"], cfg["recheck_freq"],
        cfg["adf_thresh"], cfg["pass_rate_thresh"], cfg["min_rows"],
    )


@numba.njit(cache=True)
def _adf_tstat_maxlag1(y: np.ndarray) -> float:
    """ADF t-statistic for maxlag=1, regression='c' (constant only).

    Test regression: Δy_t = α + γ·y_{t-1} + δ₁·Δy_{t-1} + ε_t
    Returns t-stat on γ (=0 under H_0: unit root). Companion p-value
    via `mackinnonp` outside numba (MacKinnon 1996 response surface).
    """
    n = len(y)
    if n < 10:
        return 0.0

    dy = y[1:] - y[:-1]
    n_obs = n - 2
    if n_obs < 5:
        return 0.0

    y_dep = dy[1:]
    y_lag = y[1:-1]
    dy_lag = dy[:-1]

    X = np.empty((n_obs, 3))
    for i in range(n_obs):
        X[i, 0] = 1.0
        X[i, 1] = y_lag[i]
        X[i, 2] = dy_lag[i]

    XtX = X.T @ X
    Xty = X.T @ y_dep

    det = np.linalg.det(XtX)
    if abs(det) < 1e-12:
        return 0.0

    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xty

    resid = y_dep - X @ beta
    rss = 0.0
    for i in range(n_obs):
        rss += resid[i] * resid[i]
    dof = n_obs - 3
    if dof <= 0:
        return 0.0
    sigma2 = rss / dof
    var_gamma = sigma2 * XtX_inv[1, 1]
    if var_gamma <= 0:
        return 0.0
    se_gamma = np.sqrt(var_gamma)
    return beta[1] / se_gamma


def adf_pvalue_fast(y: np.ndarray) -> float:
    """ADF p-value for maxlag=1, regression='c'. Numba OLS + MacKinnon p-value.

    Drop-in replacement for `statsmodels.tsa.stattools.adfuller(y, maxlag=1)[1]`
    on the maxlag=1/constant-only path. ~5-10x faster.
    """
    t_stat = _adf_tstat_maxlag1(np.ascontiguousarray(y, dtype=np.float64))
    if t_stat == 0.0:
        return 1.0
    try:
        return float(mackinnonp(t_stat, regression="c", N=1))
    except Exception:
        return 1.0


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    adf_pvalue_threshold: float,
    min_rows: int,
) -> dict | None:
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    y = p_num.loc[common].values  # numerator
    x = p_den.loc[common].values  # denominator

    # OLS hedge ratio: y = beta * x + residual
    result = scipy_stats.linregress(x, y)
    hedge_ratio = float(result.slope)
    residual = y - hedge_ratio * x

    # ADF on residual
    try:
        adf_result = adfuller(residual, maxlag=1)
        adf_pval = float(adf_result[1])
    except Exception:
        adf_pval = 1.0

    # Compute halflife on the ratio (for window estimation)
    ratio = p_num.loc[common] / p_den.loc[common]
    hl = compute_halflife(ratio)

    passed = adf_pval < adf_pvalue_threshold

    window = (
        min(max(int(hl * 0.5), 10), 252) if not np.isnan(hl) else 0
    )

    return {
        "pair": pair["pair"],
        "numerator": num,
        "denominator": den,
        "passed": passed,
        "halflife": round(hl, 2) if not np.isnan(hl) else hl,
        "window": window,
        "method": "cadf",
        "cadf_pvalue": round(adf_pval, 6),
        "hedge_ratio": round(hedge_ratio, 4),
    }


@register_stage("s1")
def cadf(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by Cointegrating ADF on OLS residual.

    Estimates hedge ratio via OLS, then tests residual for stationarity.
    More general than simple ratio ADF — allows non-1:1 relationships.

    Config:
        cadf_pvalue_threshold: float = 0.05
            Maximum ADF p-value on residual to pass.
        min_common_rows: int = 252
    """
    adf_thresh = config.get("cadf_pvalue_threshold", 0.05)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p, adf_thresh, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"cadf: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, adf_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"cadf: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)


def _process_one_rolling(
    prices: pd.DataFrame,
    pair: dict,
    window: int,
    recheck_freq: int,
    adf_thresh: float,
    pass_rate_thresh: float,
    min_rows: int,
) -> dict | None:
    """Rolling CADF per pair — re-fits β + ADF every `recheck_freq` bars over `window` lookback.

    Returns per-pair pass-rate over rolling windows. Matches the catalog B1 spec:
    "Fit β on rolling 252-day window. Recheck every 60 days. Skip pair if p > 0.05."
    """
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    y = p_num.loc[common].values
    x = p_den.loc[common].values

    # Guard against NaN/Inf — a single bad-data pair must not hang a worker
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return None
    # Guard against zero/near-zero denominators (price = 0 → infinite ratio)
    if (np.abs(x) < 1e-12).any() or (np.abs(y) < 1e-12).any():
        return None

    n = len(common)
    recheck_indices = list(range(window, n, recheck_freq))
    if len(recheck_indices) < 4:
        return None

    betas, pvals = [], []
    for t in recheck_indices:
        y_win = y[t - window:t]
        x_win = x[t - window:t]
        # Skip windows with zero variance (constant price segment)
        if x_win.std() < 1e-12 or y_win.std() < 1e-12:
            continue
        result = scipy_stats.linregress(x_win, y_win)
        beta_t = float(result.slope)
        if not np.isfinite(beta_t):
            continue
        residual = y_win - beta_t * x_win
        if not np.isfinite(residual).all():
            continue
        adf_pval = adf_pvalue_fast(residual)
        betas.append(beta_t)
        pvals.append(adf_pval)

    if len(betas) < 4:  # too many windows skipped; not enough signal
        return None

    pvals_arr = np.array(pvals)
    betas_arr = np.array(betas)
    n_windows = len(pvals_arr)
    n_pass = int((pvals_arr < adf_thresh).sum())
    pass_rate = n_pass / n_windows

    ratio = p_num.loc[common] / p_den.loc[common]
    hl = compute_halflife(ratio)

    mean_beta = float(betas_arr.mean())
    std_beta = float(betas_arr.std())
    beta_cv = (
        round(std_beta / abs(mean_beta), 4)
        if abs(mean_beta) > 1e-9
        else float("nan")
    )

    derived_window = (
        min(max(int(hl * 0.5), 10), 252) if not np.isnan(hl) else 0
    )

    return {
        "pair": pair["pair"],
        "numerator": num,
        "denominator": den,
        "passed": pass_rate >= pass_rate_thresh,
        "halflife": round(hl, 2) if not np.isnan(hl) else hl,
        "window": derived_window,
        "method": "cadf_rolling",
        "n_windows": n_windows,
        "n_pass_windows": n_pass,
        "pass_rate": round(pass_rate, 4),
        "mean_adf_pvalue": round(float(pvals_arr.mean()), 6),
        "median_adf_pvalue": round(float(np.median(pvals_arr)), 6),
        "mean_beta": round(mean_beta, 4),
        "std_beta": round(std_beta, 4),
        "beta_cv": beta_cv,
        "rolling_window": window,
        "recheck_freq": recheck_freq,
    }


@register_stage("s1")
def cadf_rolling(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Rolling Engle-Granger CADF screen — re-fits β + ADF on a rolling window.

    Catalog B1 spec (`TODO_STRATEGIES_MERGED.html#mr-b1`):
        Fit A_t = α + β × B_t + ε_t on rolling 252-day window (OLS).
        Run ADF on ε_t. Skip pair if p > 0.05.
        Re-fit every `recheck_freq` days (default 60).

    Cohort-level gate (S1 design choice, not in catalog):
        Pair PASSES if `pass_rate >= pass_rate_threshold` across rolling windows
        (default 0.5 = pair must be cointegrated in ≥ 50% of recheck windows).

    Config:
        rolling_window: int = 252           β-fit lookback (Chan QT Ch.7)
        recheck_freq: int = 60              days between fresh β + ADF
        cadf_pvalue_threshold: float = 0.05 per-window pass threshold
        pass_rate_threshold: float = 0.5    cohort-level pass-rate gate
        min_common_rows: int = 756          minimum pair history (= window + ~8 recheck windows)
        parallel: bool = True
    """
    window = config.get("rolling_window", 252)
    recheck_freq = config.get("recheck_freq", 60)
    adf_thresh = config.get("cadf_pvalue_threshold", 0.05)
    pass_rate_thresh = config.get("pass_rate_threshold", 0.5)
    min_rows = config.get("min_common_rows", 756)
    parallel = config.get("parallel", True)
    n_workers_cfg = config.get("n_workers", None)

    # Warm up numba JIT in parent process before fork (children inherit cache).
    _adf_tstat_maxlag1(np.zeros(50, dtype=np.float64))

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = n_workers_cfg or min(os.cpu_count() or 4, 6)

        # Set module-level globals BEFORE fork — children inherit via COW
        # without pickling the 100MB+ prices DataFrame.
        global _PRICES_FORK, _CONFIG_FORK
        _PRICES_FORK = prices
        _CONFIG_FORK = {
            "window": window,
            "recheck_freq": recheck_freq,
            "adf_thresh": adf_thresh,
            "pass_rate_thresh": pass_rate_thresh,
            "min_rows": min_rows,
        }

        _log(f"cadf_rolling: launching {n_workers} fork workers for {n_pairs:,} pairs")
        ctx = mp.get_context("fork")
        # chunksize tuned for ~1k-2k chunks total → low IPC overhead, even distribution
        chunksize = max(50, n_pairs // (n_workers * 200))
        try:
            with ctx.Pool(processes=n_workers, initializer=_worker_init_fork) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(_worker_process_pair, pairs, chunksize=chunksize),
                ):
                    if result is not None:
                        rows.append(result)
                    if (i + 1) % 1000 == 0 or i == n_pairs - 1:
                        _log(f"cadf_rolling: {i + 1:,}/{n_pairs:,}")
        finally:
            # Clear globals so they don't leak to future calls
            _PRICES_FORK = None
            _CONFIG_FORK = None
    else:
        for i, pair in enumerate(pairs):
            result = _process_one_rolling(
                prices, pair, window, recheck_freq,
                adf_thresh, pass_rate_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"cadf_rolling: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)


# ============================================================================
# cadf_rolling_kalman — B2 catalog spec (Kalman dynamic β + ADF on residual)
# ============================================================================
# Same scaffold as cadf_rolling above, but substitutes Kalman β_t (online
# state) for scipy.stats.linregress (per-window static OLS). Single Kalman
# pass over full pair history → residual ε_t = A_t − (β_t·B_t + α_t) at every
# bar. ADF then runs on residual within each rolling window.
#
# Conceptually different from B1's per-window OLS — Kalman β at bar t reflects
# all info up to t (online updating, no forward look-ahead).


def _process_one_rolling_kalman(
    prices: pd.DataFrame,
    pair: dict,
    window: int,
    recheck_freq: int,
    adf_thresh: float,
    pass_rate_thresh: float,
    min_rows: int,
    delta: float,
    V_eps: float,
) -> dict | None:
    """Rolling Kalman-CADF per pair (B2 spec)."""
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    A = p_num.loc[common].values
    B = p_den.loc[common].values

    # Guards (mirror static-β version)
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        return None
    if (np.abs(A) < 1e-12).any() or (np.abs(B) < 1e-12).any():
        return None

    # Single Kalman pass over full series (~1-2ms via numba)
    beta_t, alpha_t, residual_t = _kalman_beta_filter(A, B, delta, V_eps)

    if not np.isfinite(beta_t).all() or not np.isfinite(residual_t).all():
        return None

    n = len(common)
    recheck_indices = list(range(window, n, recheck_freq))
    if len(recheck_indices) < 4:
        return None

    pvals: list[float] = []
    for t in recheck_indices:
        resid_win = residual_t[t - window:t]
        if resid_win.std() < 1e-12:
            continue
        pvals.append(adf_pvalue_fast(resid_win))

    if len(pvals) < 4:
        return None

    pvals_arr = np.array(pvals)
    n_windows = len(pvals_arr)
    n_pass = int((pvals_arr < adf_thresh).sum())
    pass_rate = n_pass / n_windows

    # β stats over FULL Kalman β_t series (post burn-in)
    burn_in = min(window, max(50, len(beta_t) // 8))
    beta_post = beta_t[burn_in:]
    alpha_post = alpha_t[burn_in:]
    mean_beta = float(beta_post.mean())
    std_beta = float(beta_post.std())
    beta_cv = (
        round(std_beta / abs(mean_beta), 4)
        if abs(mean_beta) > 1e-9
        else float("nan")
    )

    # Halflife on Kalman residual (not raw ratio — more appropriate for B2)
    resid_series = pd.Series(residual_t, index=common)
    hl = compute_halflife(resid_series)

    derived_window = (
        min(max(int(hl * 0.5), 10), 252) if not np.isnan(hl) else 0
    )

    return {
        "pair": pair["pair"],
        "numerator": num,
        "denominator": den,
        "passed": pass_rate >= pass_rate_thresh,
        "halflife": round(hl, 2) if not np.isnan(hl) else hl,
        "window": derived_window,
        "method": "cadf_rolling_kalman",
        "n_windows": n_windows,
        "n_pass_windows": n_pass,
        "pass_rate": round(pass_rate, 4),
        "mean_adf_pvalue": round(float(pvals_arr.mean()), 6),
        "median_adf_pvalue": round(float(np.median(pvals_arr)), 6),
        "mean_beta": round(mean_beta, 4),
        "std_beta": round(std_beta, 4),
        "beta_cv": beta_cv,
        "mean_alpha": round(float(alpha_post.mean()), 4),
        "rolling_window": window,
        "recheck_freq": recheck_freq,
        "kalman_delta": delta,
        "kalman_V_eps": V_eps,
    }


def _worker_process_pair_kalman(pair: dict) -> dict | None:
    """Fork-pool worker for Kalman variant (mirrors _worker_process_pair)."""
    cfg = _CONFIG_FORK
    assert cfg is not None and _PRICES_FORK is not None, "globals not set"
    return _process_one_rolling_kalman(
        _PRICES_FORK, pair,
        cfg["window"], cfg["recheck_freq"],
        cfg["adf_thresh"], cfg["pass_rate_thresh"], cfg["min_rows"],
        cfg["delta"], cfg["V_eps"],
    )


@register_stage("s1")
def cadf_rolling_kalman(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Rolling Kalman-CADF screen — catalog B2 spec (Halls-Moore AAT Ch.28).

    Differs from `cadf_rolling` (static OLS β per window) by using Kalman
    dynamic β_t. Per pair: single Kalman pass over full history (~1-2ms via
    numba) → residual ε_t time series → ADF within each rolling window.

    Cohort-level gate (same as static variant):
        Pair PASSES if `pass_rate >= pass_rate_threshold` across rolling
        windows (default 0.5 = pair must show stationary Kalman residual
        in ≥ 50% of recheck windows).

    Config:
        rolling_window: int = 252           ADF window over Kalman residual
        recheck_freq: int = 60              days between ADF re-checks
        cadf_pvalue_threshold: float = 0.05 per-window pass threshold
        pass_rate_threshold: float = 0.5    cohort-level gate
        min_common_rows: int = 756          minimum pair history
        parallel: bool = True
        n_workers: int | None = None        ThreadPool/Pool size override
        kalman_delta: float = 1e-4          β/α random-walk transition variance
        kalman_V_eps: float = 1e-3          observation noise variance
    """
    window = config.get("rolling_window", 252)
    recheck_freq = config.get("recheck_freq", 60)
    adf_thresh = config.get("cadf_pvalue_threshold", 0.05)
    pass_rate_thresh = config.get("pass_rate_threshold", 0.5)
    min_rows = config.get("min_common_rows", 756)
    parallel = config.get("parallel", True)
    n_workers_cfg = config.get("n_workers", None)
    delta = config.get("kalman_delta", 1.0e-4)
    V_eps = config.get("kalman_V_eps", 1.0e-3)

    # Warm up both numba functions in parent before fork
    _adf_tstat_maxlag1(np.zeros(50, dtype=np.float64))
    _kalman_beta_filter(
        np.ones(50, dtype=np.float64),
        np.ones(50, dtype=np.float64),
        delta, V_eps,
    )

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = n_workers_cfg or min(os.cpu_count() or 4, 6)

        global _PRICES_FORK, _CONFIG_FORK
        _PRICES_FORK = prices
        _CONFIG_FORK = {
            "window": window,
            "recheck_freq": recheck_freq,
            "adf_thresh": adf_thresh,
            "pass_rate_thresh": pass_rate_thresh,
            "min_rows": min_rows,
            "delta": delta,
            "V_eps": V_eps,
        }

        _log(
            f"cadf_rolling_kalman: launching {n_workers} fork workers "
            f"for {n_pairs:,} pairs (δ={delta}, V_eps={V_eps})",
        )
        ctx = mp.get_context("fork")
        chunksize = max(50, n_pairs // (n_workers * 200))
        try:
            with ctx.Pool(processes=n_workers, initializer=_worker_init_fork) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(
                        _worker_process_pair_kalman, pairs, chunksize=chunksize,
                    ),
                ):
                    if result is not None:
                        rows.append(result)
                    if (i + 1) % 1000 == 0 or i == n_pairs - 1:
                        _log(f"cadf_rolling_kalman: {i + 1:,}/{n_pairs:,}")
        finally:
            _PRICES_FORK = None
            _CONFIG_FORK = None
    else:
        for i, pair in enumerate(pairs):
            result = _process_one_rolling_kalman(
                prices, pair, window, recheck_freq,
                adf_thresh, pass_rate_thresh, min_rows, delta, V_eps,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"cadf_rolling_kalman: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
