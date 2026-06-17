"""Tests for S2 signal methods."""
from __future__ import annotations
import numpy as np
import pandas as pd

import strategy_tester.s2_signal  # noqa: F401
from strategy_tester.registry import get_method


def _make_ou_ratio(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.ones(n)
    for i in range(1, n):
        x[i] = x[i - 1] + 0.05 * (1.0 - x[i - 1]) + 0.01 * rng.normal()
    return pd.Series(x, index=pd.bdate_range("2020-01-01", periods=n))


def test_zscore_robust_mad_shape():
    fn = get_method("s2_signal", "zscore_robust_mad")
    ratio = _make_ou_ratio()
    entries, exits = fn(ratio, window=20, entry_thresh=-2.0, exit_thresh=1.0)
    assert len(entries) == len(ratio)
    assert len(exits) == len(ratio)
    assert entries.dtype == bool
    assert entries.iloc[0] is np.bool_(False)  # shifted +1


def test_zscore_standard_shape():
    fn = get_method("s2_signal", "zscore_standard")
    ratio = _make_ou_ratio()
    entries, exits = fn(ratio, window=20, entry_thresh=-2.0, exit_thresh=1.0)
    assert len(entries) == len(ratio)
    assert entries.dtype == bool


def test_bollinger_shape():
    fn = get_method("s2_signal", "bollinger")
    ratio = _make_ou_ratio()
    entries, exits = fn(ratio, window=20, entry_thresh=-2.0, exit_thresh=0.0)
    assert len(entries) == len(ratio)
    assert entries.dtype == bool


def test_kalman_hedge_shape():
    fn = get_method("s2_signal", "kalman_hedge")
    ratio = _make_ou_ratio()
    entries, exits = fn(ratio, window=20, entry_thresh=-2.0, exit_thresh=1.0)
    assert len(entries) == len(ratio)
    assert entries.dtype == bool
