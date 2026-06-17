"""Hurst exponent screening. Chan methodology, Peters R/S analysis.

Numba-accelerated R/S core + ThreadPoolExecutor for parallel processing.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numba
import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    """Print with HH:MM:SS timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


@numba.njit(cache=True)
def _hurst_core(log_rets: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Numba-accelerated R/S computation.

    Reference: Peters, Chaos and Order in Capital Markets (1991).
    """
    n = len(log_rets)
    rs_out = np.empty(len(lags), dtype=np.float64)
    count = 0

    for li in range(len(lags)):
        lag = lags[li]
        n_chunks = n // lag
        if n_chunks == 0:
            continue
        rs_sum = 0.0
        rs_cnt = 0
        for c in range(n_chunks):
            s = c * lag
            e = s + lag
            if e > n:
                continue
            # mean
            m = 0.0
            for j in range(s, e):
                m += log_rets[j]
            m /= lag
            # cumsum of deviations — track R inline
            cum = 0.0
            mn = 0.0
            mx = 0.0
            for j in range(s, e):
                cum += log_rets[j] - m
                if cum < mn:
                    mn = cum
                if cum > mx:
                    mx = cum
            R = mx - mn
            # std ddof=1
            ss = 0.0
            for j in range(s, e):
                ss += (log_rets[j] - m) ** 2
            S = np.sqrt(ss / (lag - 1)) if lag > 1 else 0.0
            if S > 0.0:
                rs_sum += R / S
                rs_cnt += 1
        if rs_cnt > 0:
            rs_out[count] = rs_sum / rs_cnt
            count += 1

    return rs_out[:count]


def _hurst_rs(series: pd.Series, num_lags: int = 50) -> float:
    """Hurst exponent via R/S analysis (numba-accelerated core)."""
    vals = np.log(series.values[1:] / series.values[:-1])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50:
        return float("nan")

    max_lag = len(vals) // 2
    lags = np.unique(
        np.logspace(0.5, np.log10(max_lag), num_lags).astype(np.int64)
    )
    lags = lags[(lags > 0) & (lags < len(vals))]
    if len(lags) < 5:
        return float("nan")

    rs_values = _hurst_core(vals.astype(np.float64), lags)
    if len(rs_values) < 5:
        return float("nan")

    log_lags = np.log10(lags[: len(rs_values)].astype(np.float64))
    log_rs = np.log10(rs_values)
    from scipy.stats import linregress

    result = linregress(log_lags, log_rs)
    return float(result.slope)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    hurst_thresh: float,
    min_rows: int,
) -> dict | None:
    """Process one pair (runs in thread)."""
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None
    common = prices[num].dropna().index.intersection(
        prices[den].dropna().index
    )
    if len(common) < min_rows:
        return None

    ratio = prices[num].loc[common] / prices[den].loc[common]
    hl = compute_halflife(ratio)
    h = _hurst_rs(ratio)

    passed = not np.isnan(h) and h < hurst_thresh
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
        "method": "chan_hurst",
        "hurst_exponent": round(h, 4) if not np.isnan(h) else h,
    }


@register_stage("s1")
def chan_hurst(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by Hurst exponent on ratio. H < threshold = MR."""
    hurst_thresh = config.get("hurst_threshold", 0.50)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p, hurst_thresh, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"chan_hurst: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, hurst_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"chan_hurst: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
