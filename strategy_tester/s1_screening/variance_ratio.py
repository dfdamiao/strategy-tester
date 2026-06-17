"""Variance Ratio test for mean reversion. Lo & MacKinlay (1988).

VR(q) = Var(q-period returns) / (q * Var(1-period returns))
VR(q) < 1 → anti-persistent (mean-reverting)
VR(q) = 1 → random walk
VR(q) > 1 → persistent (trending)

We test multiple holding periods q = [2, 5, 10, 20] and require
the median VR < threshold to pass. This is more robust than a single q.

Reference:
    Lo & MacKinlay (1988), "Stock market prices do not follow random walks:
    Evidence from a simple specification test", RFS 1(1).
    Chan, Algorithmic Trading (2013) Ch.2 — VR as MR diagnostic.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage

VR_PERIODS = [2, 5, 10, 20]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _variance_ratio(prices: np.ndarray, q: int) -> float:
    """Compute VR(q) from a price series (Lo & MacKinlay 1988).

    Uses overlap-corrected estimator for efficiency.
    """
    n = len(prices)
    if n < q + 10:
        return np.nan

    log_prices = np.log(prices)
    # 1-period returns
    ret1 = np.diff(log_prices)
    # q-period returns
    ret_q = log_prices[q:] - log_prices[:-q]

    var1 = np.var(ret1, ddof=1)
    var_q = np.var(ret_q, ddof=1)

    if var1 == 0 or np.isnan(var1):
        return np.nan

    # VR(q) = Var(q-ret) / (q * Var(1-ret))
    # Adjusted for overlap: multiply by (n-1) / (n-q) correction
    nq = len(ret_q)
    n1 = len(ret1)
    vr = (var_q / var1) * (n1 / (q * nq))
    return float(vr)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    vr_threshold: float,
    min_rows: int,
    periods: list[int],
) -> dict | None:
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    ratio = p_num.loc[common] / p_den.loc[common]
    ratio_clean = ratio.dropna()
    if len(ratio_clean) < max(periods) + 20:
        return None

    hl = compute_halflife(ratio)

    # Compute VR at multiple periods
    vr_values = [_variance_ratio(ratio_clean.values, q) for q in periods]
    vr_valid = [v for v in vr_values if not np.isnan(v)]

    if not vr_valid:
        return None

    median_vr = float(np.median(vr_valid))

    # VR < 1 → mean-reverting. Pass if median VR < threshold.
    passed = median_vr < vr_threshold

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
        "method": "variance_ratio",
        "median_vr": round(median_vr, 4),
    }


@register_stage("s1")
def variance_ratio(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by Variance Ratio test on ratio.

    VR(q) < 1 indicates mean reversion. Median across multiple
    holding periods q = [2, 5, 10, 20] must be below threshold.

    Config:
        vr_threshold: float = 0.95
            Maximum median VR to pass (< 1 = mean-reverting).
        min_common_rows: int = 252
    """
    vr_thresh = config.get("vr_threshold", 0.95)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)
    periods = config.get("vr_periods", VR_PERIODS)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p,
                    vr_thresh, min_rows, periods,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"variance_ratio: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, vr_thresh, min_rows, periods,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"variance_ratio: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
