"""Hurst exponent trend screening. H > threshold = persistent/trending.

References: Peters, Chaos and Order in Capital Markets (1991).
Caveat: Lo (1991) — finite-sample bias in R/S estimator.
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
from strategy_tester.s1_screening.chan_hurst import _hurst_rs


def _log(msg: str) -> None:
    """Print with HH:MM:SS timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    hurst_thresh: float,
    min_rows: int,
) -> dict | None:
    """Process one pair/single for Hurst trend screening."""
    num, den = pair["numerator"], pair["denominator"]
    is_single = pair.get("asset_type") == "single"

    if num not in prices.columns:
        return None
    if not is_single and den not in prices.columns:
        return None

    if is_single:
        common = prices[num].dropna().index
        series = prices[num].loc[common]
    else:
        common = prices[num].dropna().index.intersection(
            prices[den].dropna().index
        )
        series = prices[num].loc[common] / prices[den].loc[common]

    if len(common) < min_rows:
        return None

    hl = compute_halflife(series)
    h = _hurst_rs(series)

    passed = not np.isnan(h) and h > hurst_thresh
    window = 0  # set by S2 optimizer for trend strategies

    return {
        "pair": pair["pair"],
        "numerator": num,
        "denominator": den,
        "passed": passed,
        "halflife": round(hl, 2) if not np.isnan(hl) else hl,
        "window": window,
        "method": "chan_hurst_trend",
        "hurst_exponent": round(h, 4) if not np.isnan(h) else h,
    }


@register_stage("s1")
def chan_hurst_trend(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by Hurst exponent. H > threshold = trending."""
    hurst_thresh = config.get("hurst_trend_threshold", 0.50)
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
                    _log(f"chan_hurst_trend: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, hurst_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"chan_hurst_trend: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
