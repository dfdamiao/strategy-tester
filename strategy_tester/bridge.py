"""Bridge — join Pipeline S2/S3/S4 outputs for build_portfolio.py.

Pipeline S3 methods return validation metrics only (mean_test_sharpe,
degradation, etc.) — they do NOT carry forward the backtest parameters
(halflife, window, entry_thresh, exit_thresh, stop_pct) which live in S2.
Pipeline S4 adds tier/passed but also lacks backtest params.

build_portfolio.py needs the full row: backtest params (S2) + validation
metrics (S3) + tier (S4). This module bridges that gap.

When multiple S3 methods are used (e.g. ["wfa_expanding", "cpcv"]),
the pipeline concatenates — you get N rows per pair (one per method).
Three resolution strategies are available to collapse to 1 row per pair.

Usage:
    result = pipe.run(prices, pairs, config, stop_after="s4")
    enriched = bridge_to_portfolio(result, mode="strictest")
    # → DataFrame with S2 params + S3 metrics + S4 tier, 1 row per pair

Academic references:
    - Bailey, Borwein & LdP (2017): single param set per pair for CPCV
    - Pardo §8.3, §9.1: multi-method validation strengthens confidence
    - Paleologo APM Ch.4: numerator deduplication
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from strategy_tester.interfaces import PipelineResult

# Tier ordering: lower rank = better
TIER_ORDER = ["TOP_TIER", "SECOND_TIER", "LAST_TIER", "REJECT"]


def _tier_rank(tier: str) -> int:
    """Lower = better. Unknown/REJECT = 3."""
    try:
        return TIER_ORDER.index(tier)
    except ValueError:
        return 3


def _downgrade(tier: str) -> str:
    """Downgrade tier by one level."""
    rank = _tier_rank(tier)
    return TIER_ORDER[min(rank + 1, 3)]


def _resolve_multi_s3(
    s3_df: pd.DataFrame,
    mode: Literal["strictest", "agreement", "any_pass"],
) -> pd.DataFrame:
    """Resolve multiple S3 rows per pair to 1 row per pair.

    Parameters
    ----------
    s3_df : DataFrame
        Concatenated S3 output (may have N rows per pair, one per
        val_method).
    mode : str
        Resolution strategy:

        ``"strictest"``
            Pair must pass ALL S3 methods used. Tier = worst across
            methods. Conservative, fewest false positives.
            (Pardo §8.3: multi-method agreement = strongest evidence.)

        ``"agreement"``
            Agreement-based labels matching stage3_merge.py logic:
            - CONFIRMED: passes all methods → tier = worst of all
            - FRAGILE_EDGE: passes some but not all → downgrade best
              passing tier by 1
            - REJECT: fails all methods
            Preserves nuance about which methods agreed.
            (Generalises the 2-method 3A/3B merge to N methods.)

        ``"any_pass"``
            Pair passes if ANY S3 method passes. Tier = best across
            methods. Most survivors, weakest gate. Useful for
            exploratory runs or when methods test orthogonal properties.

    Returns
    -------
    DataFrame
        One row per pair with columns:
        - All S3_REQUIRED columns (mean_test_sharpe uses the selected
          method's value)
        - val_methods: comma-separated list of methods used
        - agreement: label (only in "agreement" mode)
        - Per-method Sharpe columns: sharpe_{method_name}
    """
    if s3_df.empty:
        return s3_df

    methods = s3_df["val_method"].unique().tolist()

    # Single method — no resolution needed
    if len(methods) <= 1:
        return s3_df.copy()

    pairs = s3_df["pair"].unique()
    rows = []

    for pair in pairs:
        pair_rows = s3_df[s3_df["pair"] == pair]
        n_methods = len(pair_rows)
        n_passed = int(pair_rows["passed"].sum())

        # Collect per-method Sharpes
        method_sharpes = {}
        for _, r in pair_rows.iterrows():
            method_sharpes[f"sharpe_{r['val_method']}"] = r[
                "mean_test_sharpe"
            ]

        # Base row: pick the one with best mean_test_sharpe for
        # column inheritance (numerator, denominator, etc.)
        best_idx = pair_rows["mean_test_sharpe"].idxmax()
        base = pair_rows.loc[best_idx].to_dict()

        if mode == "strictest":
            passed = n_passed == n_methods
            if passed:
                # Worst Sharpe across methods
                base["mean_test_sharpe"] = pair_rows[
                    "mean_test_sharpe"
                ].min()
                # Worst degradation (highest value)
                base["degradation"] = pair_rows["degradation"].max()
            else:
                base["passed"] = False

        elif mode == "agreement":
            if n_passed == n_methods:
                agreement = "CONFIRMED"
                base["mean_test_sharpe"] = pair_rows[
                    "mean_test_sharpe"
                ].min()
                base["degradation"] = pair_rows["degradation"].max()
                passed = True
            elif n_passed > 0:
                agreement = "FRAGILE_EDGE"
                # Use best passing method's Sharpe, downgraded
                passing = pair_rows[pair_rows["passed"]]
                best_pass_idx = passing["mean_test_sharpe"].idxmax()
                base = passing.loc[best_pass_idx].to_dict()
                passed = True
            else:
                agreement = "REJECT"
                passed = False

            base["agreement"] = agreement

        elif mode == "any_pass":
            passed = n_passed > 0
            if passed:
                # Best Sharpe across passing methods
                passing = pair_rows[pair_rows["passed"]]
                best_pass_idx = passing["mean_test_sharpe"].idxmax()
                base = passing.loc[best_pass_idx].to_dict()

        base["passed"] = passed
        base["val_methods"] = ",".join(
            pair_rows["val_method"].tolist()
        )
        base["n_methods_passed"] = n_passed
        base["n_methods_total"] = n_methods
        base.update(method_sharpes)
        rows.append(base)

    return pd.DataFrame(rows)


def bridge_to_portfolio(
    result: PipelineResult,
    mode: Literal["strictest", "agreement", "any_pass"] = "strictest",
    deduplicate_numerator: bool = True,
) -> pd.DataFrame:
    """Bridge Pipeline S2/S3/S4 outputs into build_portfolio.py format.

    Parameters
    ----------
    result : PipelineResult
        Output from ``Pipeline.run(stop_after="s4")``. Must contain
        stages s2, s3, and s4.
    mode : str
        S3 multi-method resolution strategy. See ``_resolve_multi_s3``.
    deduplicate_numerator : bool
        If True, keep only the best pair per numerator (highest
        mean_test_sharpe). Paleologo APM Ch.4.

    Returns
    -------
    DataFrame
        One row per pair with columns from S2 (backtest params),
        S3 (validation metrics), and S4 (tier). Ready for
        build_portfolio.py consumption.

    Raises
    ------
    ValueError
        If required stages (s2, s3, s4) are missing from result.
    """
    # --- Extract stage outputs ---
    for stage in ("s2", "s3", "s4"):
        if stage not in result.stages:
            raise ValueError(
                f"Stage {stage!r} missing from PipelineResult. "
                f"Available: {list(result.stages.keys())}. "
                f"Run with stop_after='s4' or later."
            )

    s2_df = result.stages["s2"]["result"]
    s3_df = result.stages["s3"]["result"]
    s4_df = result.stages["s4"]["result"]

    # --- Step 1: Resolve multi-S3 to 1 row per pair ---
    s3_resolved = _resolve_multi_s3(s3_df, mode=mode)

    # --- Step 2: Extract S2 backtest params (1 row per pair) ---
    # S2 may have multiple rows per pair (multiple param_ranks).
    # Take the one that S3 validated (best OOS Sharpe).
    s2_params_cols = [
        "pair", "numerator", "denominator",
        "halflife", "window",
        "entry_thresh", "exit_thresh", "stop_pct", "slope_min",
    ]
    # Keep only columns that exist
    s2_params_cols = [c for c in s2_params_cols if c in s2_df.columns]

    s2_passed = s2_df[s2_df["passed"]].copy()
    if "oos_sharpe" in s2_passed.columns:
        s2_passed = s2_passed.sort_values(
            "oos_sharpe", ascending=False,
        )
    s2_deduped = s2_passed.drop_duplicates(
        subset="pair", keep="first",
    )[s2_params_cols]

    # --- Step 3: Extract S4 tier (already 1 row per pair) ---
    # Add any S4-specific columns (sig_method, t_stat, etc.)
    s4_extra = [
        c for c in s4_df.columns
        if c not in ("pair", "numerator", "denominator", "passed")
    ]
    s4_slim = s4_df[["pair"] + s4_extra].copy()
    # Rename tier to s4_tier to avoid clash with S3
    if "tier" in s4_slim.columns:
        s4_slim = s4_slim.rename(columns={"tier": "s4_tier"})

    # --- Step 4: Join S2 params + S3 metrics + S4 tier ---
    # S3 resolved has: pair, numerator, denominator, mean_test_sharpe,
    #   degradation, passed, val_methods, ...
    # S2 deduped has: pair, halflife, window, entry_thresh, ...
    # S4 slim has: pair, s4_tier, ...

    enriched = s3_resolved.merge(s2_deduped, on="pair", how="left",
                                  suffixes=("", "_s2"))
    enriched = enriched.merge(s4_slim, on="pair", how="left",
                               suffixes=("", "_s4"))

    # Resolve numerator/denominator: prefer S2 (always present)
    for col in ("numerator", "denominator"):
        s2_col = f"{col}_s2"
        if s2_col in enriched.columns:
            enriched[col] = enriched[col].fillna(enriched[s2_col])
            enriched = enriched.drop(columns=[s2_col])

    # --- Step 5: Apply S4 gate ---
    # Only keep pairs that passed S4
    s4_passed = s4_df[s4_df["passed"]]["pair"].tolist()
    enriched = enriched[enriched["pair"].isin(s4_passed)].copy()

    # Use S4 tier as the authoritative tier
    if "s4_tier" in enriched.columns:
        enriched["tier"] = enriched["s4_tier"]

    # --- Step 6: Map tier to result column (build_portfolio compat) ---
    enriched["result"] = enriched["tier"].apply(
        lambda t: "GOOD" if t in ("TOP_TIER", "SECOND_TIER")
        else "MARGINAL" if t == "LAST_TIER"
        else "REJECT"
    )
    enriched["deduplicated"] = False

    # --- Step 7: Numerator deduplication (Paleologo APM Ch.4) ---
    if deduplicate_numerator and not enriched.empty:
        enriched = enriched.sort_values(
            "mean_test_sharpe", ascending=False,
        )
        dup_mask = enriched.duplicated(subset="numerator", keep="first")
        enriched.loc[dup_mask, "deduplicated"] = True
        enriched.loc[dup_mask, "dedup_reason"] = "numerator_overlap"

    # --- Step 8: Drop internal columns ---
    drop_cols = [
        c for c in enriched.columns
        if c.startswith("_") or c == "s4_tier"
    ]
    enriched = enriched.drop(columns=drop_cols, errors="ignore")

    return enriched.reset_index(drop=True)
