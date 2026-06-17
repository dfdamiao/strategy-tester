"""Kaufman Efficiency Ratio screening. Low median ER = noisy/mean-reverting.

Reference: Kaufman, TSM 6e Ch.17 — ER is a rolling metric computed at
every bar with a short lookback (default 10 bars). We take the median
of rolling ER over the evaluation period as the screening statistic.

ThreadPoolExecutor for parallel pair processing.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    """Print with HH:MM:SS timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _rolling_er(series: np.ndarray, n: int) -> np.ndarray:
    """Compute rolling Efficiency Ratio per Kaufman TSM Ch.17.

    ER[t] = |price[t] - price[t-n]| / sum(|price[i] - price[i-1]|, i=t-n+1..t)

    Returns array of ER values (NaN for first n bars).
    """
    length = len(series)
    er = np.full(length, np.nan)
    abs_diff = np.abs(np.diff(series))  # |price[i] - price[i-1]|, length-1

    for t in range(n, length):
        direction = abs(series[t] - series[t - n])
        volatility = abs_diff[t - n : t].sum()
        er[t] = direction / volatility if volatility > 0 else 1.0

    return er


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    er_thresh: float,
    er_window: int,
    min_rows: int,
) -> dict | None:
    """Process one pair for Kaufman ER screening."""
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

    # Rolling ER per Kaufman TSM Ch.17
    vals = ratio.values.astype(np.float64)
    er_arr = _rolling_er(vals, er_window)
    valid_er = er_arr[~np.isnan(er_arr)]
    if len(valid_er) == 0:
        return None

    median_er = float(np.median(valid_er))
    passed = median_er < er_thresh  # Low ER = mean-reverting
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
        "method": "kaufman_er",
        "efficiency_ratio": round(median_er, 4),
    }


@register_stage("s1")
def kaufman_er(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by rolling Efficiency Ratio. Low median ER = mean-reverting.

    Config keys:
        er_threshold (0.30): median ER must be below this to pass
        er_window (10): Kaufman ER lookback in bars (TSM Ch.17 default)
        min_common_rows (252): minimum data overlap
    """
    er_thresh = config.get("er_threshold", 0.30)
    er_window = config.get("er_window", 10)
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
                    er_thresh, er_window, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"kaufman_er: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, er_thresh, er_window, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"kaufman_er: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
