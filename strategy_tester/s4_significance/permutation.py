"""One-sided parametric z-test on fold Sharpe. (NOT a permutation test.)

History: this module was named ``permutation`` but never permuted anything.
It draws no shuffled signals and resamples no returns; it compares the mean
fold Sharpe against the analytic 95th percentile of a N(0, std) null. The
S4 interface only receives S3 summary statistics (mean/std/n of fold
Sharpes), so a true shuffled-signal permutation test is impossible at this
stage without an interface change. Renamed honestly to ``sr_ztest``
(2026-07-28); ``permutation`` remains as a deprecated alias so existing
configs keep working.
"""

from __future__ import annotations

import warnings

import pandas as pd

from strategy_tester.registry import register_stage

Z_95_ONE_SIDED = 1.6449
_STD_FLOOR = 0.1  # retained from the original implementation


@register_stage("s4")
def sr_ztest(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """Pass if mean fold Sharpe > 1.6449 * max(std, 0.1): one-sided 95%
    z-test against a N(0, std) null. Analytic, no Monte Carlo noise."""
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = row["mean_test_sharpe"]
        std = row.get("std_test_sharpe", 0.5)
        n = row["n_test_periods"]
        if std == 0 or n < 2:
            p95 = 0.0
            passed = False
        else:
            p95 = Z_95_ONE_SIDED * max(std, _STD_FLOOR)
            passed = sr > p95
        rows.append(
            {
                "pair": row["pair"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "passed": passed,
                "tier": "TOP_TIER" if passed else "REJECT",
                "ztest_p95": round(p95, 4),
                "ztest_passed": passed,
                "sig_method": "sr_ztest",
            }
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["pair", "numerator", "denominator", "passed", "tier"]
        )
    )


@register_stage("s4")
def permutation(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """Deprecated alias for :func:`sr_ztest`. This was never a permutation
    test; use ``sr_ztest``."""
    warnings.warn(
        "s4 method 'permutation' is a misnomer (it is a parametric z-test, "
        "not a permutation test) and is deprecated; use 'sr_ztest'.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sr_ztest(s3_result, **config)
