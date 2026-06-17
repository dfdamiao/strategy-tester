"""Tests for stage interface validation."""
from __future__ import annotations
import pandas as pd
import pytest
from strategy_tester.interfaces import validate_interface


def test_s1_output_valid():
    df = pd.DataFrame({
        "pair": ["A/B"], "numerator": ["A"],
        "denominator": ["B"], "passed": [True],
        "halflife": [30.0], "window": [15],
        "method": ["chan_halflife"],
    })
    validate_interface(df, "s1")  # should not raise


def test_s1_output_missing_column():
    df = pd.DataFrame({"pair": ["A/B"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_interface(df, "s1")


def test_s2_output_valid():
    df = pd.DataFrame({
        "pair": ["A/B"], "numerator": ["A"],
        "denominator": ["B"], "halflife": [30.0],
        "window": [15], "entry_thresh": [-2.0],
        "exit_thresh": [1.5], "stop_pct": [0.1],
        "slope_min": [0.0], "is_sharpe": [1.2],
        "is_penalized_sharpe": [1.0], "is_trades": [20],
        "oos_sharpe": [0.8], "oos_trades": [5],
        "passed": [True], "signal_method": ["zscore_robust_mad"],
        "optim_method": ["grid_search"],
    })
    validate_interface(df, "s2")  # should not raise


def test_unknown_stage_raises():
    df = pd.DataFrame({"pair": ["A/B"]})
    with pytest.raises(ValueError, match="Unknown stage"):
        validate_interface(df, "s99")
