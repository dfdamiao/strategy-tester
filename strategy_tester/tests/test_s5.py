"""Tests for S5 portfolio methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategy_tester.s5_portfolio  # noqa: F401
from strategy_tester.registry import get_method


def _make_s4_result() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "A/B",
                "numerator": "A",
                "denominator": "B",
                "passed": True,
                "tier": "TOP_TIER",
            },
            {
                "pair": "C/D",
                "numerator": "C",
                "denominator": "D",
                "passed": True,
                "tier": "SECOND_TIER",
            },
        ]
    )


def _make_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(
        {
            "A": 100 + np.cumsum(rng.normal(0, 1, 500)),
            "B": 100 + np.cumsum(rng.normal(0, 1, 500)),
            "C": 50 + np.cumsum(rng.normal(0, 0.5, 500)),
            "D": 50 + np.cumsum(rng.normal(0, 0.5, 500)),
        },
        index=idx,
    )


def test_equal_weight() -> None:
    fn = get_method("s5", "equal_weight")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result
    assert abs(result["weights"]["A/B"] - 0.5) < 0.01
    assert "sharpe" in result
    assert "equity_curve" in result
    assert result["portfolio_method"] == "equal_weight"


def test_inverse_vol() -> None:
    fn = get_method("s5", "inverse_vol")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=0.01)


def test_risk_parity() -> None:
    fn = get_method("s5", "risk_parity")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_hrp() -> None:
    fn = get_method("s5", "hrp")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_sharpe_weighted() -> None:
    fn = get_method("s5", "sharpe_weighted")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_half_kelly() -> None:
    fn = get_method("s5", "half_kelly")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_dynamic_1k() -> None:
    fn = get_method("s5", "dynamic_1k")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_tier_based() -> None:
    fn = get_method("s5", "tier_based")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result


def test_handcraft_carver() -> None:
    fn = get_method("s5", "handcraft_carver")
    result = fn(_make_prices(), _make_s4_result())
    assert "weights" in result
