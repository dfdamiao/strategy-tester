"""AR(1) halflife + ADF screening. Chan, Algorithmic Trading Ch.2.

ThreadPoolExecutor for parallel pair processing (avoids fork deadlocks
on macOS with numpy/scipy/statsmodels).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    """Print with HH:MM:SS timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    min_hl: float,
    max_hl: float,
    adf_thresh: float,
    min_rows: int,
) -> dict | None:
    """Process one pair for halflife + ADF screening."""
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    ratio = p_num.loc[common] / p_den.loc[common]
    hl = compute_halflife(ratio)

    adf_pval = np.nan
    if not np.isnan(hl):
        try:
            adf_result = adfuller(ratio.dropna(), maxlag=1)
            adf_pval = float(adf_result[1])
        except Exception:
            adf_pval = 1.0

    passed = (
        not np.isnan(hl)
        and min_hl <= hl <= max_hl
        and not np.isnan(adf_pval)
        and adf_pval < adf_thresh
    )
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
        "method": "chan_halflife",
        "adf_pvalue": (
            round(adf_pval, 6) if not np.isnan(adf_pval) else adf_pval
        ),
    }


@register_stage("s1")
def chan_halflife(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by AR(1) halflife + ADF test on ratio."""
    min_hl = config.get("min_halflife", 2)
    max_hl = config.get("max_halflife", 756)
    adf_thresh = config.get("adf_pvalue_threshold", 0.05)
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
                    min_hl, max_hl, adf_thresh, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"chan_halflife: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, min_hl, max_hl, adf_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"chan_halflife: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
