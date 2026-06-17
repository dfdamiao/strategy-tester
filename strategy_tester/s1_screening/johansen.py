"""Johansen cointegration test. Johansen (1988).

Tests for cointegration in a multivariate framework using the trace
and maximum eigenvalue statistics. More powerful than Engle-Granger
(CADF) for detecting cointegrating relationships in 2+ asset systems.

For pairs: tests if rank(cointegration) >= 1, meaning at least one
stationary linear combination exists between numerator and denominator.

Reference:
    Johansen (1988), "Statistical analysis of cointegration vectors",
    Journal of Economic Dynamics and Control 12(2-3).
    Chan, Algorithmic Trading (2013) Ch.2 — Johansen for pairs.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    significance: int,
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

    # Build 2-column matrix for Johansen
    data = np.column_stack([
        p_num.loc[common].values,
        p_den.loc[common].values,
    ])

    # Johansen test: det_order=-1 means no deterministic trend in data
    # k_ar_diff=1: 1 lag in VECM (standard for daily prices)
    try:
        result = coint_johansen(data, det_order=-1, k_ar_diff=1)
    except Exception:
        return {
            "pair": pair["pair"],
            "numerator": num,
            "denominator": den,
            "passed": False,
            "halflife": np.nan,
            "window": 0,
            "method": "johansen",
            "trace_stat_r0": np.nan,
            "trace_crit_r0": np.nan,
        }

    # Trace test for r=0 (no cointegration) vs r>=1
    # significance: 0 = 90%, 1 = 95%, 2 = 99%
    trace_stat = float(result.lr1[0])   # trace stat for r=0
    trace_crit = float(result.cvt[0, significance])  # critical value

    # Reject r=0 if trace_stat > critical value → cointegrated
    passed = trace_stat > trace_crit

    ratio = p_num.loc[common] / p_den.loc[common]
    hl = compute_halflife(ratio)
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
        "method": "johansen",
        "trace_stat_r0": round(trace_stat, 4),
        "trace_crit_r0": round(trace_crit, 4),
    }


@register_stage("s1")
def johansen(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by Johansen cointegration trace test.

    Tests H0: no cointegration (r=0) vs H1: at least one
    cointegrating vector (r>=1). More powerful than CADF
    for detecting cointegrating relationships.

    Config:
        johansen_significance: int = 1
            Critical value index: 0=90%, 1=95%, 2=99%.
        min_common_rows: int = 252
    """
    significance = config.get("johansen_significance", 1)  # 95%
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p, significance, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"johansen: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, significance, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"johansen: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
