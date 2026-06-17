"""Carver 2-sigma CI on Sharpe. Carver (2015)."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


@register_stage("s4")
def carver_2sigma(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """CI lower = SR - 2*(std/sqrt(n)) > 0. Carver (2015)."""
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        std = row.get("std_test_sharpe", 0.5)
        n = row["n_test_periods"]
        # Guard: need variance estimate and >=2 folds for CI to be meaningful
        if std == 0 or n < 2:
            ci_lower = 0.0
            ci_upper = float(sr)
            passed = False
        else:
            se = std / (n**0.5)
            ci_lower = sr - 2 * se
            ci_upper = sr + 2 * se
            passed = ci_lower > 0
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "sig_method": "carver_2sigma",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
