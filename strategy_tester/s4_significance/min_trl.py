"""Minimum Track Record Length. Bailey & LdP (2012)."""
from __future__ import annotations

import pandas as pd
from scipy.stats import norm

from strategy_tester.registry import register_stage


@register_stage("s4")
def min_trl(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """Compute MinTRL in years. Informational — does not gate."""
    z_alpha = norm.ppf(0.975)  # 95% CI
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        skew = 0.0
        kurtosis = 3.0
        if abs(sr) < 1e-9:
            years: float = float("inf")
        else:
            denom_sq = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
            if denom_sq <= 0:
                years = float("inf")
            else:
                n_obs = 1 + denom_sq * (z_alpha / sr) ** 2
                years = float(n_obs / 252)
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": True,  # Informational only
            "tier": "TOP_TIER",
            "min_trl_years": (
                round(years, 2) if years != float("inf") else years
            ),
            "sig_method": "min_trl",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
