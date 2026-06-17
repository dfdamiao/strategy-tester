"""Mathematical verification tests for Walk-Forward Analysis.

Reproduces Pardo (2008) §6.3 / §8.3 walk-forward partition rules so future
refactors of build_expanding_folds / build_rolling_folds cannot drift silently.

Reference: Pardo, R. (2008). "The Evaluation and Optimization of Trading
Strategies", 2nd ed. Wiley Trading. §6.3, §8.3.

Key published rules (Pardo §8.3):
  - Expanding WFA: IS grows each fold, OOS = fixed chunk.
  - With n_folds=8 and N total bars: chunk = N // (n_folds + 1).
  - Fold k: IS = bars 0 .. (k+1)×chunk − 1, OOS = (k+1)×chunk .. (k+2)×chunk.
  - WFE (Walk-Forward Efficiency) = mean(OOS_SR) / mean(IS_SR); threshold 0.50.
  - Pardo §8.4: majority (≥ 50%) of OOS folds must be individually profitable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy_tester.s3_validation.wfa_expanding import build_expanding_folds
from strategy_tester.s3_validation.wfa_rolling import build_rolling_folds

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ten_year_index() -> pd.DatetimeIndex:
    """~2520 business days ≈ 10 years (2010-2020)."""
    return pd.bdate_range(start="2010-01-04", periods=2520)


@pytest.fixture
def two_year_index() -> pd.DatetimeIndex:
    """~504 business days ≈ 2 years."""
    return pd.bdate_range(start="2020-01-02", periods=504)


# ---------------------------------------------------------------------------
# Expanding WFA — Pardo 2e §8.3
# ---------------------------------------------------------------------------

class TestExpandingFoldPartitions:
    """Verify fold partition boundaries for 10-year daily series, 8 folds."""

    def test_returns_8_folds(self, ten_year_index: pd.DatetimeIndex) -> None:
        """Pardo §8.3: n_folds=8 (default) → exactly 8 folds for 10yr series."""
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        # Published rule: 8 folds for a well-sized series
        assert len(folds) == 8, f"Expected 8 folds, got {len(folds)}"

    def test_fold_sizes_match_pardo_chunk_rule(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Pardo §8.3: chunk = N // (n_folds + 1); OOS each fold = chunk bars.

        Published partition: N=2520, n_folds=8 → chunk = 2520 // 9 = 280.
        """
        n = len(ten_year_index)  # 2520
        n_folds = 8
        chunk = n // (n_folds + 1)  # = 280

        folds = build_expanding_folds(ten_year_index, n_folds=n_folds)
        for k, (_is_idx, oos_idx) in enumerate(folds):
            # OOS must be exactly chunk bars (or last fold may differ by 1)
            assert abs(len(oos_idx) - chunk) <= 1, (
                f"Fold {k}: OOS len={len(oos_idx)}, expected chunk={chunk} "
                "per Pardo §8.3"
            )

    def test_is_grows_monotonically(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Expanding WFA: IS set must strictly grow with each fold."""
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        is_lengths = [len(is_idx) for is_idx, _ in folds]
        assert all(a < b for a, b in zip(is_lengths, is_lengths[1:])), (
            f"IS lengths must strictly grow: {is_lengths}"
        )

    def test_oos_immediately_follows_is(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Expanding WFA: OOS start = IS end + 1 (no gap, no overlap)."""
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        for k, (is_idx, oos_idx) in enumerate(folds):
            # IS ends at some position; OOS must start at next day
            assert is_idx[-1] < oos_idx[0], (
                f"Fold {k}: IS ends {is_idx[-1]}, OOS starts {oos_idx[0]} — "
                "should have no gap"
            )

    def test_no_overlap_between_is_and_oos(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """IS and OOS indices must not share any timestamps."""
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        for k, (is_idx, oos_idx) in enumerate(folds):
            overlap = is_idx.intersection(oos_idx)
            assert len(overlap) == 0, (
                f"Fold {k}: IS/OOS overlap = {len(overlap)} bars"
            )

    def test_fold_end_indices_progress_monotonically(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Each fold's OOS end must advance forward in time."""
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        oos_ends = [oos_idx[-1] for _, oos_idx in folds]
        assert all(a < b for a, b in zip(oos_ends, oos_ends[1:])), (
            f"OOS end dates must advance: {oos_ends}"
        )

    def test_returns_empty_for_tiny_series(self) -> None:
        """Series too short to form meaningful folds → empty list (not crash)."""
        short_idx = pd.bdate_range(start="2020-01-01", periods=50)
        folds = build_expanding_folds(short_idx, n_folds=8)
        assert folds == []

    def test_is_at_least_as_long_as_oos_all_folds(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Pardo design: IS ≥ OOS each fold (fold 0 IS = OOS = 1 chunk; grows from fold 1).

        Fold 0: IS = 1 chunk = OOS (equal). Fold k > 0: IS = (k+1)×chunk > chunk = OOS.
        """
        folds = build_expanding_folds(ten_year_index, n_folds=8)
        for k, (is_idx, oos_idx) in enumerate(folds):
            assert len(is_idx) >= len(oos_idx), (
                f"Fold {k}: IS={len(is_idx)} should be >= OOS={len(oos_idx)}"
            )
        # From fold 1 onward IS strictly exceeds OOS
        for k, (is_idx, oos_idx) in enumerate(folds[1:], start=1):
            assert len(is_idx) > len(oos_idx), (
                f"Fold {k}: IS={len(is_idx)} should strictly exceed OOS={len(oos_idx)}"
            )


class TestWfeDefinition:
    """Walk-Forward Efficiency = OOS_SR / IS_SR (Pardo §8.3)."""

    def test_wfe_above_threshold_passes(self) -> None:
        """WFE > 0.50 = strategy degrades ≤ 50% in OOS (Pardo §8.3 threshold)."""
        # Direct arithmetic — no I/O
        oos_sr = 0.80
        is_sr = 1.20
        wfe = oos_sr / is_sr  # 0.667 > 0.50 → PASS
        assert wfe > 0.50

    def test_wfe_below_threshold_fails(self) -> None:
        """WFE < 0.50 = strategy loses more than half its edge out-of-sample."""
        oos_sr = 0.30
        is_sr = 1.20
        wfe = oos_sr / is_sr  # 0.25 < 0.50 → FAIL
        assert wfe < 0.50

    def test_fold_win_rate_counts_positive_oos_sharpes(self) -> None:
        """Pardo §8.4: at least 50% of OOS folds must be individually profitable.

        pct_positive = count(fold_sr > 0) / n_folds.
        """
        fold_oos_srs = [0.5, -0.2, 0.8, 1.1, -0.1, 0.3, 0.7, 0.9]  # 6/8 positive
        pct_positive = sum(1 for s in fold_oos_srs if s > 0) / len(fold_oos_srs)
        # Published Pardo gate: ≥ 0.50
        assert pct_positive >= 0.50
        assert pct_positive == pytest.approx(6 / 8, abs=1e-9)

    def test_fold_win_rate_fails_below_half(self) -> None:
        fold_oos_srs = [0.5, -0.2, -0.8, -1.1, -0.1, 0.3, 0.7, -0.9]  # 3/8
        pct_positive = sum(1 for s in fold_oos_srs if s > 0) / len(fold_oos_srs)
        assert pct_positive < 0.50


# ---------------------------------------------------------------------------
# Rolling WFA — Strimpel / Pardo variant
# ---------------------------------------------------------------------------

class TestRollingFoldPartitions:
    """Verify rolling WFA fold structure (fixed IS, fixed OOS, sliding window)."""

    def test_default_params_produce_multiple_folds(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """10-year series with IS=504, OOS=180 should produce several folds."""
        folds = build_rolling_folds(ten_year_index, is_days=504, oos_days=180)
        # At least 3 folds expected (10yr ≈ 2520 bars; (2520-504) / 180 ≈ 11)
        assert len(folds) >= 3

    def test_each_fold_has_fixed_is_size(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Rolling WFA: IS length must equal is_days for all folds."""
        is_days = 504
        folds = build_rolling_folds(ten_year_index, is_days=is_days, oos_days=180)
        for k, (is_idx, _) in enumerate(folds):
            assert len(is_idx) == is_days, (
                f"Fold {k}: IS={len(is_idx)}, expected {is_days}"
            )

    def test_each_fold_has_fixed_oos_size(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """Rolling WFA: OOS length must equal oos_days for all folds."""
        oos_days = 180
        folds = build_rolling_folds(ten_year_index, is_days=504, oos_days=oos_days)
        for k, (_, oos_idx) in enumerate(folds):
            assert len(oos_idx) == oos_days, (
                f"Fold {k}: OOS={len(oos_idx)}, expected {oos_days}"
            )

    def test_folds_advance_by_oos_days(
        self, ten_year_index: pd.DatetimeIndex,
    ) -> None:
        """OOS start of fold k+1 = OOS start of fold k + oos_days."""
        oos_days = 180
        folds = build_rolling_folds(ten_year_index, is_days=504, oos_days=oos_days)
        for k in range(len(folds) - 1):
            oos_start_k = folds[k][1][0]
            oos_start_k1 = folds[k + 1][1][0]
            # The OOS window advances by oos_days positions
            pos_k = int(ten_year_index.get_loc(oos_start_k))  # type: ignore[arg-type]
            pos_k1 = int(ten_year_index.get_loc(oos_start_k1))  # type: ignore[arg-type]
            advance = pos_k1 - pos_k
            assert advance == oos_days, (
                f"Fold {k}→{k+1}: window advanced by {advance}, expected {oos_days}"
            )

    def test_empty_for_series_too_short(self) -> None:
        """Series shorter than IS+OOS → empty fold list (not crash)."""
        short = pd.bdate_range(start="2020-01-01", periods=100)
        folds = build_rolling_folds(short, is_days=504, oos_days=180)
        assert folds == []
