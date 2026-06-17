"""t-test on fold Sharpe ratios. Chan (2013) Ch.3."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


@register_stage("s4")
def t_test(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """t-stat = mean_SR / (std_SR / sqrt(n)). Threshold default 2.0."""
    threshold = config.get("t_stat_threshold", 2.0)
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        std = row.get("std_test_sharpe", 0.5)
        n = row["n_test_periods"]
        if std == 0 or n < 2:
            t_stat = 0.0
        else:
            t_stat = sr / (std / (n**0.5))
        passed = t_stat > threshold
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "t_stat": round(t_stat, 4),
            "t_passed": passed,
            "sig_method": "t_test",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
