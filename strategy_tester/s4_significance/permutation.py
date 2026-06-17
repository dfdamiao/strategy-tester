"""Parametric z-test on fold Sharpe. (NOT White's Reality Check.)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


@register_stage("s4")
def permutation(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """Pass if SR > 95th percentile of N(0, std) null. Parametric z-test."""
    n_perms = config.get("permutation_n", 1000)
    random_state = config.get("random_state", 42)
    rows = []
    rng = np.random.default_rng(random_state)
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        std = row.get("std_test_sharpe", 0.5)
        n = row["n_test_periods"]
        # Guard: need variance estimate and >=2 folds
        if std == 0 or n < 2:
            p95 = 0.0
            passed = False
        else:
            # Parametric null — N(0, std), NOT permutation of returns
            null_sharpes = rng.normal(0, max(std, 0.1), n_perms)
            p95 = float(np.percentile(null_sharpes, 95))
            passed = sr > p95
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "perm_p95": round(p95, 4),
            "perm_passed": passed,
            "sig_method": "permutation",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
