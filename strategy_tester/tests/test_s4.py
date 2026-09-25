"""Tests for S4 significance methods."""

from __future__ import annotations

import pandas as pd
import pytest

import strategy_tester.s4_significance  # noqa: F401
from strategy_tester.registry import get_method
from strategy_tester.interfaces import validate_interface


def _make_s3_result(val_method: str = "cpcv") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "A/B",
                "numerator": "A",
                "denominator": "B",
                "mean_test_sharpe": 1.5,
                "std_test_sharpe": 0.3,
                "n_test_periods": 45,
                "baseline_sharpe": 1.8,
                "degradation": 0.17,
                "passed": True,
                "val_method": val_method,
            }
        ]
    )


def _make_wfa_s3_result() -> pd.DataFrame:
    return _make_s3_result("wfa_expanding").assign(wfe=0.65)


def test_psr_s4_format() -> None:
    fn = get_method("s4", "psr")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")
    assert "psr_stat" in result.columns


def test_dsr_requires_n_trials() -> None:
    fn = get_method("s4", "dsr")
    with pytest.raises(KeyError, match="n_trials"):
        fn(_make_s3_result())


def test_dsr_with_n_trials() -> None:
    fn = get_method("s4", "dsr")
    result = fn(_make_s3_result(), n_trials=1716)
    validate_interface(result, "s4")


def test_wfe_rejects_cpcv() -> None:
    fn = get_method("s4", "wfe")
    with pytest.raises(ValueError, match="WFE requires WFA"):
        fn(_make_s3_result("cpcv"))


def test_wfe_accepts_wfa() -> None:
    fn = get_method("s4", "wfe")
    result = fn(_make_wfa_s3_result())
    validate_interface(result, "s4")


def test_t_test_s4_format() -> None:
    fn = get_method("s4", "t_test")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_min_trl_s4_format() -> None:
    fn = get_method("s4", "min_trl")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_min_trl_deannualizes_sharpe() -> None:
    """mean_test_sharpe is ANNUALIZED; the Bailey & LdP MinTRL formula takes the
    Sharpe at the native (daily) frequency. Feeding the annualized value made
    every MinTRL ~135x too short: SR=1.0 must read ~3.85 years at the module's
    two-sided 95% z (1.96), not ~0.03."""
    fn = get_method("s4", "min_trl")
    out = fn(_make_s3_result().assign(mean_test_sharpe=1.0))
    years = float(out.loc[0, "min_trl_years"])
    assert years == pytest.approx(3.85, abs=0.05), years


def test_carver_2sigma_s4_format() -> None:
    fn = get_method("s4", "carver_2sigma")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_permutation_s4_format() -> None:
    # deprecated alias: still registered, still interface-valid, but warns
    fn = get_method("s4", "permutation")
    with pytest.warns(DeprecationWarning, match="sr_ztest"):
        result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_sr_ztest_s4_format_and_analytic_threshold() -> None:
    """The honest name for what this method always was: a one-sided
    parametric z-test on fold Sharpe against N(0, std). The 95th percentile
    is analytic (1.6449 * std floored at 0.1), no Monte Carlo noise."""
    fn = get_method("s4", "sr_ztest")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")
    row = result.iloc[0]
    assert row["sig_method"] == "sr_ztest"
    # std_test_sharpe = 0.3 -> threshold = 1.6449 * 0.3
    assert row["ztest_p95"] == pytest.approx(1.6449 * 0.3, abs=1e-3)
    assert bool(row["passed"]) is (1.5 > 1.6449 * 0.3)


def test_sr_ztest_guards_degenerate_inputs() -> None:
    fn = get_method("s4", "sr_ztest")
    df = _make_s3_result()
    df.loc[0, "std_test_sharpe"] = 0.0
    result = fn(df)
    assert bool(result.iloc[0]["passed"]) is False


def test_drawdown_s4_format() -> None:
    fn = get_method("s4", "drawdown")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")
