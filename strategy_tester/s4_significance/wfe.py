"""Walk-Forward Efficiency. Pardo 2e §8."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


@register_stage("s4")
def wfe(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """WFE = OOS Sharpe / IS Sharpe. Only valid for WFA outputs."""
    # Routing constraint: WFE only from WFA, not CPCV
    val_methods = set(s3_result["val_method"].unique())
    wfa_methods = {"wfa_expanding", "wfa_rolling"}
    if not (val_methods & wfa_methods):
        raise ValueError(
            "WFE requires WFA output (CPCV has no IS/OOS "
            "in Pardo sense). S3 methods found: "
            f"{sorted(val_methods)}"
        )
    threshold = config.get("wfe_threshold", 0.50)
    # Filter to WFA rows only
    wfa_rows = s3_result[
        s3_result["val_method"].isin(wfa_methods)
        & s3_result["passed"]
    ]
    rows = []
    for _, row in wfa_rows.iterrows():
        w = row.get("wfe", 0.0)
        if pd.isna(w):
            w = 0.0
        passed = w > threshold
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "wfe_stat": round(w, 4),
            "wfe_passed": passed,
            "sig_method": "wfe",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
