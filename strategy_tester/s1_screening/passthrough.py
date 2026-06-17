"""Passthrough S1 screening — marks all items as passed.

For trend-following strategies where statistical pre-screening
is not recommended (Murphy 1999, Carver 2015, Chan 2008).
The S2 optimization + OOS gate serves as the effective screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


@register_stage("s1")
def passthrough(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Pass all items through. S2 optimization is the real screen.

    References:
        Murphy, Technical Analysis (1999) — apply MA to everything
        Carver, Systematic Trading (2015) Ch.5 — no trend pre-filter
        Chan, Quantitative Trading (2008) — OOS validation is the screen
    """
    min_rows = config.get("min_common_rows", 252)
    rows: list[dict] = []

    for pair in pairs:
        num = pair["numerator"]
        den = pair["denominator"]
        is_single = pair.get("asset_type") == "single"

        if num not in prices.columns:
            continue
        if not is_single and den not in prices.columns:
            continue

        if is_single:
            common = prices[num].dropna().index
            series = prices[num].loc[common]
        else:
            common = (
                prices[num].dropna().index
                .intersection(prices[den].dropna().index)
            )
            if len(common) < min_rows:
                continue
            series = prices[num].loc[common] / prices[den].loc[common]

        if len(series) < min_rows:
            continue

        hl = compute_halflife(series)

        # Window: use halflife-derived if available, else default
        # For trend strategies: window comes from grid_ma sweep
        # For RSI/other: use default_window from config (14 standard)
        default_window = config.get("default_window", 20)
        if not np.isnan(hl) and hl > 0:
            window = min(max(int(hl * 0.5), 10), 252)
        else:
            window = default_window

        rows.append({
            "pair": pair["pair"],
            "numerator": num,
            "denominator": den,
            "passed": True,
            "halflife": (
                round(hl, 2) if not np.isnan(hl) else float("nan")
            ),
            "window": window,
            "method": "passthrough",
        })

    return pd.DataFrame(rows)
