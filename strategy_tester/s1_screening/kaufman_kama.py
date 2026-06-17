"""KAMA-based screening. Kaufman, Trading Systems and Methods 6e Ch.17.

Computes Kaufman Adaptive Moving Average on price ratio, then screens
by mean Efficiency Ratio and ratio-KAMA crossing frequency.

A pair is mean-reverting if:
  - mean ER is low (ratio is consistently noisy, not trending)
  - ratio crosses KAMA frequently (actively oscillating around trend)

ThreadPoolExecutor for parallel pair processing.
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
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


@numba.njit(cache=True)
def _kama_core(
    prices: np.ndarray,
    period: int,
    fast_sc: float,
    slow_sc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute KAMA and per-bar ER. Kaufman TSM 6e Ch.17.

    Returns:
        kama: KAMA values (NaN for first `period` bars)
        er: Efficiency Ratio per bar (NaN for first `period` bars)
    """
    n = len(prices)
    kama = np.full(n, np.nan)
    er_arr = np.full(n, np.nan)

    if n <= period:
        return kama, er_arr

    # Initialize KAMA at the first valid bar
    kama[period] = prices[period]

    for i in range(period, n):
        # Direction = |price[i] - price[i-period]|
        direction = abs(prices[i] - prices[i - period])

        # Volatility = sum of |price[j] - price[j-1]| over period
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(prices[j] - prices[j - 1])

        # Efficiency Ratio
        if volatility > 0:
            er = direction / volatility
        else:
            er = 0.0
        er_arr[i] = er

        # Smoothing Constant: SC = [ER × (fast - slow) + slow]²
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA update
        if i == period:
            kama[i] = prices[i]
        else:
            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

    return kama, er_arr


def _count_crossings(ratio: np.ndarray, kama: np.ndarray) -> int:
    """Count how many times ratio crosses KAMA."""
    valid = ~np.isnan(kama)
    r = ratio[valid]
    k = kama[valid]
    if len(r) < 2:
        return 0
    above = r > k
    crossings = int(np.sum(above[1:] != above[:-1]))
    return crossings


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    kama_period: int,
    fast_n: int,
    slow_n: int,
    mean_er_thresh: float,
    min_crossings_yr: float,
    min_rows: int,
) -> dict | None:
    """Process one pair for KAMA screening."""
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    common = prices[num].dropna().index.intersection(
        prices[den].dropna().index
    )
    if len(common) < min_rows:
        return None

    ratio = prices[num].loc[common] / prices[den].loc[common]
    ratio_vals = ratio.values.astype(np.float64)

    hl = compute_halflife(ratio)

    # KAMA parameters
    fast_sc = 2.0 / (fast_n + 1)
    slow_sc = 2.0 / (slow_n + 1)

    kama, er_arr = _kama_core(ratio_vals, kama_period, fast_sc, slow_sc)

    # Mean ER (exclude NaN warmup)
    valid_er = er_arr[~np.isnan(er_arr)]
    mean_er = float(np.mean(valid_er)) if len(valid_er) > 0 else 1.0

    # Crossings per year
    crossings = _count_crossings(ratio_vals, kama)
    n_years = len(common) / 252
    crossings_yr = crossings / n_years if n_years > 0 else 0

    # KAMA slope: annualized drift of KAMA / mean ratio
    valid_kama = kama[~np.isnan(kama)]
    if len(valid_kama) > 10:
        kama_drift = abs(valid_kama[-1] - valid_kama[0]) / len(valid_kama)
        kama_slope = kama_drift * 252 / np.mean(ratio_vals)
    else:
        kama_slope = 0.0

    # Gate: low mean ER AND sufficient crossings
    passed = mean_er < mean_er_thresh and crossings_yr >= min_crossings_yr

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
        "method": "kaufman_kama",
        "mean_er": round(mean_er, 4),
        "crossings_yr": round(crossings_yr, 1),
        "kama_slope": round(kama_slope, 4),
    }


@register_stage("s1")
def kaufman_kama(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by KAMA behavior on ratio.

    Kaufman, TSM 6e Ch.17: KAMA adapts speed via ER.
    - mean_er < threshold → consistently noisy (MR candidate)
    - crossings_per_year >= minimum → actively oscillating

    Config:
        kama_period: lookback for ER (default 10)
        kama_fast: fast EMA period (default 2)
        kama_slow: slow EMA period (default 30)
        kama_mean_er_thresh: max mean ER to pass (default 0.15)
        kama_min_crossings_yr: min crossings/year (default 4.0)
    """
    kama_period = config.get("kama_period", 10)
    fast_n = config.get("kama_fast", 2)
    slow_n = config.get("kama_slow", 30)
    mean_er_thresh = config.get("kama_mean_er_thresh", 0.15)
    min_crossings_yr = config.get("kama_min_crossings_yr", 4.0)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p,
                    kama_period, fast_n, slow_n,
                    mean_er_thresh, min_crossings_yr, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"kaufman_kama: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair,
                kama_period, fast_n, slow_n,
                mean_er_thresh, min_crossings_yr, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"kaufman_kama: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
