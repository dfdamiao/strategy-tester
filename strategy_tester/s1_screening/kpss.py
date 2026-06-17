"""KPSS stationarity test. Kwiatkowski, Phillips, Schmidt & Shin (1992).

Complementary to ADF: H0 = stationary (opposite of ADF's H0 = unit root).
Pair passes if KPSS fails to reject H0 (p-value >= threshold), confirming
the ratio is stationary → mean-reverting.

Using both ADF (reject unit root) and KPSS (fail to reject stationarity)
provides a more robust stationarity conclusion than either alone.

Reference:
    Kwiatkowski et al. (1992), "Testing the null hypothesis of stationarity
    against the alternative of a unit root", Journal of Econometrics 54.
    Chan, Algorithmic Trading (2013) Ch.2 — complementary stationarity tests.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss as kpss_test

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage



def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    kpss_pvalue_threshold: float,
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

    ratio = p_num.loc[common] / p_den.loc[common]
    hl = compute_halflife(ratio)

    # KPSS test: H0 = stationary. We WANT to fail to reject (p >= threshold).
    kpss_pval = np.nan
    try:
        # regression='c' tests level stationarity (not trend)
        # KPSS p-values bounded [0.01, 0.10] — suppress interpolation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _stat, pval, _lags, _crit = kpss_test(
                ratio.dropna().values, regression="c", nlags="auto",
            )
        kpss_pval = float(pval)
    except Exception:
        kpss_pval = 0.0  # failed test → treat as non-stationary

    # Pass if KPSS fails to reject stationarity (p >= threshold)
    passed = not np.isnan(kpss_pval) and kpss_pval >= kpss_pvalue_threshold

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
        "method": "kpss",
        "kpss_pvalue": round(kpss_pval, 6),
    }


@register_stage("s1")
def kpss(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Screen pairs by KPSS stationarity test on ratio.

    KPSS H0 = stationary. Pair passes if we FAIL to reject H0
    (p-value >= threshold), confirming the ratio is stationary.

    Config:
        kpss_pvalue_threshold: float = 0.05
            Minimum p-value to pass (fail to reject stationarity).
        min_common_rows: int = 252
    """
    kpss_thresh = config.get("kpss_pvalue_threshold", 0.05)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p, kpss_thresh, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"kpss: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair, kpss_thresh, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"kpss: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
