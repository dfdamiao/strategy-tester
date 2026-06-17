"""Combined trend screening — ER + Hurst AND gate."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage, get_method


@register_stage("s1")
def trend_combined(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Run ER trend + Hurst trend as AND gate (both must pass)."""
    er_fn = get_method("s1", "kaufman_er_trend")
    hurst_fn = get_method("s1", "chan_hurst_trend")

    er_df = er_fn(prices, pairs, **config)
    hurst_df = hurst_fn(prices, pairs, **config)

    if er_df.empty:
        return pd.DataFrame(
            columns=[
                "pair", "numerator", "denominator", "passed",
                "halflife", "window", "method",
                "efficiency_ratio", "hurst_exponent",
            ]
        )

    merged = er_df[
        ["pair", "numerator", "denominator",
         "halflife", "window", "efficiency_ratio"]
    ].copy()
    merged = merged.rename(columns={})

    # ER passed
    merged["er_passed"] = er_df["passed"].values

    # Hurst passed
    if not hurst_df.empty:
        merged = merged.merge(
            hurst_df[["pair", "passed", "hurst_exponent"]].rename(
                columns={"passed": "hurst_passed"}
            ),
            on="pair", how="left",
        )
    else:
        merged["hurst_passed"] = False
        merged["hurst_exponent"] = float("nan")

    # Fill NaN passes as False
    for col in ["er_passed", "hurst_passed"]:
        merged[col] = merged[col].fillna(False)

    # AND gate — both must pass
    merged["passed"] = merged["er_passed"] & merged["hurst_passed"]

    merged["method"] = "trend_combined"
    return merged[
        ["pair", "numerator", "denominator", "passed",
         "halflife", "window", "method",
         "efficiency_ratio", "hurst_exponent"]
    ]
