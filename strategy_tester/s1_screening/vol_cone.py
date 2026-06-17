"""Volatility Cone filter. Sinclair, Volatility Trading 2e (2013) Ch.7.

Secondary filter applied on top of a primary S1 screen.
Rejects pairs whose current realized volatility is in an extreme regime
(top or bottom percentile of historical vol distribution).

Extreme vol → unreliable mean reversion. Very low vol → too small moves
to trade profitably. Very high vol → regime break, cointegration unstable.

The "cone" concept: at each lookback window, compute historical vol,
build a percentile distribution (cone), and check if current vol is
within the acceptable band.

Reference:
    Sinclair, Volatility Trading 2e (2013) Ch.7 — Volatility cones.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage

# Default lookback windows for the cone (trading days)
CONE_WINDOWS = [21, 63, 126, 252]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    lower_pct: float,
    upper_pct: float,
    vol_window: int,
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
    returns = ratio.pct_change().dropna()

    if len(returns) < vol_window + 50:
        return None

    # Rolling realized vol (annualized)
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < 50:
        return None

    # Current vol = last observation
    current_vol = float(rolling_vol.iloc[-1])

    # Historical percentile of current vol
    vol_percentile = float(
        (rolling_vol < current_vol).mean() * 100
    )

    # Pass if current vol is within acceptable band
    passed = lower_pct <= vol_percentile <= upper_pct

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
        "method": "vol_cone",
        "current_vol": round(current_vol, 4),
        "vol_percentile": round(vol_percentile, 1),
    }


@register_stage("s1")
def vol_cone(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Filter pairs by volatility cone — reject extreme vol regimes.

    Secondary filter. Current realized vol must fall between the
    lower and upper percentile of the historical vol distribution.

    Config:
        vol_cone_lower_pct: float = 10.0
            Reject if current vol < this percentile (too quiet).
        vol_cone_upper_pct: float = 90.0
            Reject if current vol > this percentile (too wild).
        vol_cone_window: int = 63
            Rolling vol lookback in bars (~3 months).
        min_common_rows: int = 252
    """
    lower_pct = config.get("vol_cone_lower_pct", 10.0)
    upper_pct = config.get("vol_cone_upper_pct", 90.0)
    vol_window = config.get("vol_cone_window", 63)
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
                    lower_pct, upper_pct, vol_window, min_rows,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"vol_cone: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair,
                lower_pct, upper_pct, vol_window, min_rows,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"vol_cone: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
