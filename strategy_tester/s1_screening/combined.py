"""Combined S1 screening — AND/OR gate across all sub-methods."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage, get_method


@register_stage("s1")
def chan_combined(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Run halflife + hurst + ER as AND gate (all must pass)."""
    hl_fn = get_method("s1", "chan_halflife")
    hurst_fn = get_method("s1", "chan_hurst")
    er_fn = get_method("s1", "kaufman_er")

    hl_df = hl_fn(prices, pairs, **config)
    hurst_df = hurst_fn(prices, pairs, **config)
    er_df = er_fn(prices, pairs, **config)

    # Merge on pair
    combine_mode = config.get("s1_combine_mode", "intersect")

    merged = hl_df[["pair", "numerator", "denominator",
                     "halflife", "window", "passed"]].copy()
    merged = merged.rename(columns={"passed": "hl_passed"})

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

    if not er_df.empty:
        merged = merged.merge(
            er_df[["pair", "passed", "efficiency_ratio"]].rename(
                columns={"passed": "er_passed"}
            ),
            on="pair", how="left",
        )
    else:
        merged["er_passed"] = False
        merged["efficiency_ratio"] = float("nan")

    # Fill NaN passes as False
    for col in ["hl_passed", "hurst_passed", "er_passed"]:
        merged[col] = merged[col].fillna(False)

    if combine_mode == "intersect":
        merged["passed"] = (
            merged["hl_passed"]
            & merged["hurst_passed"]
            & merged["er_passed"]
        )
    else:  # union
        merged["passed"] = (
            merged["hl_passed"]
            | merged["hurst_passed"]
            | merged["er_passed"]
        )

    merged["method"] = "chan_combined"
    return merged[["pair", "numerator", "denominator", "passed",
                    "halflife", "window", "method",
                    "hurst_exponent", "efficiency_ratio"]]
