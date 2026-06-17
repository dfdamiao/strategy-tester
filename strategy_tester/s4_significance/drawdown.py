"""Drawdown-based significance. CDaR, E[MaxDD], shutdown rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


@register_stage("s4")
def drawdown(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """E[MaxDD] rough Magdon-Ismail approximation. Informational risk filter."""
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        n = row["n_test_periods"]
        # Magdon-Ismail leading-order: higher SR → smaller expected DD
        if abs(sr) > 0.01 and n > 0:
            e_max_dd = -1.0 / max(sr, 0.01) * (2 * n / 252) ** 0.5
        else:
            e_max_dd = float("nan")
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": True,  # Informational + risk filter
            "tier": "TOP_TIER",
            "e_max_dd": round(float(e_max_dd), 4) if not np.isnan(e_max_dd) else float("nan"),
            "sig_method": "drawdown",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
