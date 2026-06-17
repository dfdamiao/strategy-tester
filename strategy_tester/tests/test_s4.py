"""Tests for S4 significance methods."""
from __future__ import annotations

import pandas as pd
import pytest

import strategy_tester.s4_significance  # noqa: F401
from strategy_tester.registry import get_method
from strategy_tester.interfaces import validate_interface


def _make_s3_result(val_method: str = "cpcv") -> pd.DataFrame:
    return pd.DataFrame([{
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
    }])


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


def test_carver_2sigma_s4_format() -> None:
    fn = get_method("s4", "carver_2sigma")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_permutation_s4_format() -> None:
    fn = get_method("s4", "permutation")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")


def test_drawdown_s4_format() -> None:
    fn = get_method("s4", "drawdown")
    result = fn(_make_s3_result())
    validate_interface(result, "s4")
