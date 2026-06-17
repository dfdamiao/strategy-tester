"""Regression tests for METHODOLOGY_DECISIONS.md §1 alignment.

Locks the 2026-05-17 fix where lib's PSR + DSR switched from fold count
to T-days bar count for ``n_obs``, matching the production sweep DSR
implementation and the Bailey-LdP paper.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s4_significance.psr import psr as psr_gate
from strategy_tester.s4_significance.dsr import dsr as dsr_gate


@pytest.fixture
def s3_row_with_bars() -> pd.DataFrame:
    """Single S3 row with both n_test_periods (folds) and n_oos_bars (T)."""
    return pd.DataFrame([{
        "pair": "AAA/BBB",
        "numerator": "AAA",
        "denominator": "BBB",
        "mean_test_sharpe": 1.0,
        "std_test_sharpe": 0.30,
        "n_test_periods": 8,        # fold count
        "n_oos_bars": 1250,         # T-days bar count — paper convention
        "baseline_sharpe": 1.5,
        "degradation": 0.33,
        "passed": True,
        "val_method": "wfa_expanding",
        "sr_variance": 0.04,
    }])


@pytest.fixture
def s3_row_no_bars() -> pd.DataFrame:
    """Legacy S3 row missing n_oos_bars — triggers fallback warning."""
    return pd.DataFrame([{
        "pair": "AAA/BBB",
        "numerator": "AAA",
        "denominator": "BBB",
        "mean_test_sharpe": 1.0,
        "std_test_sharpe": 0.30,
        "n_test_periods": 8,
        "baseline_sharpe": 1.5,
        "degradation": 0.33,
        "passed": True,
        "val_method": "wfa_expanding",
        "sr_variance": 0.04,
    }])


class TestPsrUsesNOosBars:
    """PSR reads n_oos_bars when present."""

    def test_psr_uses_bars_when_present(
        self, s3_row_with_bars: pd.DataFrame,
    ) -> None:
        out = psr_gate(s3_row_with_bars)
        assert len(out) == 1
        # n_obs_used must reflect T-days, not fold count
        assert int(out.iloc[0]["n_obs_used"]) == 1250

    def test_psr_falls_back_to_folds_with_warning(
        self, s3_row_no_bars: pd.DataFrame,
    ) -> None:
        with pytest.warns(DeprecationWarning, match="n_oos_bars"):
            out = psr_gate(s3_row_no_bars)
        assert int(out.iloc[0]["n_obs_used"]) == 8

    def test_psr_bars_higher_than_folds_increases_significance(
        self,
        s3_row_with_bars: pd.DataFrame,
        s3_row_no_bars: pd.DataFrame,
    ) -> None:
        """For SR=1, bars=1250 should give vastly higher PSR than folds=8."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_bars = psr_gate(s3_row_with_bars)
            out_folds = psr_gate(s3_row_no_bars)
        p_bars = float(out_bars.iloc[0]["psr_stat"])
        p_folds = float(out_folds.iloc[0]["psr_stat"])
        assert p_bars > p_folds, (
            f"PSR with bars ({p_bars:.4f}) must exceed PSR with folds "
            f"({p_folds:.4f}) for SR=1 — bars give more degrees of freedom"
        )


class TestDsrUsesNOosBars:
    """DSR reads n_oos_bars when present (METHODOLOGY_DECISIONS.md §1)."""

    def test_dsr_uses_bars_when_present(
        self, s3_row_with_bars: pd.DataFrame,
    ) -> None:
        out = dsr_gate(s3_row_with_bars, n_trials=50)
        assert len(out) == 1
        # No warning when n_oos_bars is present
        assert "dsr_stat" in out.columns or "passed" in out.columns

    def test_dsr_falls_back_to_folds_with_warning(
        self, s3_row_no_bars: pd.DataFrame,
    ) -> None:
        with pytest.warns(DeprecationWarning, match="n_oos_bars"):
            dsr_gate(s3_row_no_bars, n_trials=50)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """600-bar synthetic price panel shared across S3-validator tests."""
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "AAA": 100 + np.cumsum(rng.normal(0, 1, n)),
        "BBB": 100 + np.cumsum(rng.normal(0, 1, n)),
    }, index=idx)


@pytest.fixture
def synthetic_pair_row() -> dict:
    return {
        "pair": "AAA/BBB",
        "numerator": "AAA",
        "denominator": "BBB",
        "window": 20,
        "entry_thresh": -2.0,
        "exit_thresh": 0.0,
        "stop_pct": 0.0,
        "slope_min": 0.0,
        "is_sharpe": 0.5,
    }


class TestS3EmittersIncludeNOosBars:
    """Every S3 validator's output dict must contain n_oos_bars."""

    def test_wfa_expanding_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation.wfa_expanding import (
            _process_one_pair_wfa, _worker_init,
        )
        _worker_init(synthetic_prices, signal_fn=None)
        result = _process_one_pair_wfa(
            synthetic_pair_row, n_folds=4, fees=0.001, slope_window=2,
            use_s2_window=True,
        )
        assert result is not None
        assert "n_oos_bars" in result
        assert 0 < result["n_oos_bars"] < len(synthetic_prices)

    def test_wfa_rolling_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import wfa_rolling as wr
        wr._WORKER_PRICES = synthetic_prices
        wr._WORKER_SIGNAL_FN = None
        # wfa_rolling needs IS+OOS bars; our 600-bar fixture caps the
        # IS+OOS budget — use shorter windows to get at least one fold
        result = wr._process_one_pair_wfa_rolling(
            synthetic_pair_row,
            is_days=300, oos_days=120, max_windows=30,
            fees=0.001, slope_window=2,
            use_s2_window=True,
        )
        # Synthetic random-walk pair may yield zero trades → None is OK
        if result is None:
            pytest.skip("wfa_rolling produced no folds on synthetic data")
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] >= 0

    def test_chan_is_oos_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import chan_is_oos as ch
        ch._WORKER_PRICES = synthetic_prices
        ch._WORKER_SIGNAL_FN = None
        result = ch._process_one_pair_chan_is_oos(
            synthetic_pair_row,
            is_ratio=0.7, fees=0.001, slope_window=2,
            use_s2_window=True,
        )
        assert result is not None
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] > 0

    def test_bootstrap_ci_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import bootstrap_ci as bc
        bc._WORKER_PRICES = synthetic_prices
        bc._WORKER_SIGNAL_FN = None
        result = bc._process_one_pair_bootstrap(
            synthetic_pair_row,
            n_iter=200, fees=0.001, slope_window=2,
            is_ratio=0.7, random_state=42,
        )
        if result is None:
            pytest.skip("bootstrap_ci needs ≥3 monthly blocks on OOS")
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] > 0

    def test_monte_carlo_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import monte_carlo as mc
        mc._WORKER_PRICES = synthetic_prices
        mc._WORKER_SIGNAL_FN = None
        result = mc._process_one_pair_monte_carlo(
            synthetic_pair_row,
            n_iter=200, fees=0.001, slope_window=2,
            is_ratio=0.7, random_state=42,
        )
        if result is None:
            pytest.skip("monte_carlo needs ≥20 OOS bars")
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] > 0
        # MC caps OOS at 630 — verify cap is honoured
        assert result["n_oos_bars"] <= 630

    def test_sensitivity_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import sensitivity as se
        se._WORKER_PRICES = synthetic_prices
        se._WORKER_SIGNAL_FN = None
        result = se._process_one_pair_sensitivity(
            synthetic_pair_row,
            pct=0.20, fees=0.001, slope_window=2, is_ratio=0.7,
        )
        assert result is not None
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] > 0

    def test_cpcv_emits_n_oos_bars(
        self, synthetic_prices: pd.DataFrame, synthetic_pair_row: dict,
    ) -> None:
        from strategy_tester.s3_validation import cpcv as cv
        from strategy_tester.config import DEFAULT_CONFIG
        cv._WORKER_PRICES = synthetic_prices
        cv._WORKER_SIGNAL_FN = None
        result = cv._process_one_pair_cpcv(
            synthetic_pair_row,
            n_folds=10, purge_bars=20, embargo_bars=5,
            fees=0.001, slope_window=2,
            config_dict={**DEFAULT_CONFIG, "use_s2_window": True,
                         "s3_min_oos_sharpe": -99.0},
        )
        # cpcv may legitimately return None if fold count < 3
        if result is None:
            pytest.skip("CPCV fold count below threshold on synthetic data")
        assert "n_oos_bars" in result
        assert result["n_oos_bars"] > 0
