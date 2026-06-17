"""Probabilistic Sharpe Ratio. Bailey & LdP (2012).

Methodology lock 2026-05-17 (METHODOLOGY_DECISIONS.md §1):
``n_obs`` = total OOS bar count (T-days) per the paper's worked example.
Falls back to ``n_test_periods`` (fold count) with a deprecation warning
when ``n_oos_bars`` is absent from the upstream S3 output — old artifacts
predate the column.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.metrics import psr_stat


def _safe_float(val: object, default: float) -> float:
    """Coerce a row cell to float, falling back to ``default`` on None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if np.isnan(f) else f


def _resolve_n_obs(row: pd.Series) -> int:
    """Return T-days bar count; warn + fall back to fold count if missing."""
    n_bars = _safe_float(row.get("n_oos_bars"), float("nan"))
    if np.isnan(n_bars) or n_bars <= 0:
        warnings.warn(
            "lib.s4_significance.psr: row missing 'n_oos_bars'; falling "
            "back to 'n_test_periods' (fold count). Re-run S3 to populate "
            "the bar-count column. See METHODOLOGY_DECISIONS.md §1.",
            DeprecationWarning,
            stacklevel=3,
        )
        return int(_safe_float(row.get("n_test_periods"), 2.0))
    return int(n_bars)


@register_stage("s4")
def psr(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """PSR > 0.95 gate. Bailey & LdP (2012).

    Uses OOS bar count (T-days) as ``n_obs`` per the paper. Fold-count
    fallback is preserved for legacy callers but emits a deprecation
    warning — see METHODOLOGY_DECISIONS.md §1.
    """
    # Skew/kurtosis convention: Fisher skew, regular (NOT excess) kurtosis.
    # Defaults (0, 3) assume normal returns; override via row columns when
    # populated by S2.
    skew_default = float(config.get("skew", 0.0))
    kurt_default = float(config.get("kurtosis", 3.0))
    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = _safe_float(row["mean_test_sharpe"], 0.0)
        n_obs = _resolve_n_obs(row)
        skew = _safe_float(row.get("skew"), skew_default)
        kurtosis = _safe_float(row.get("kurtosis"), kurt_default)
        p = psr_stat(sr, max(n_obs, 2), skew, kurtosis)
        passed = p > 0.95
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "psr_stat": round(p, 4),
            "psr_passed": passed,
            "n_obs_used": int(n_obs),
            "sig_method": "psr",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
