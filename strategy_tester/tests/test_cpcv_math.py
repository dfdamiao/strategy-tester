"""Mathematical verification tests for CPCV partition rules.

Reproduces López de Prado AFML (2018) Ch.12 combinatorial purged cross-
validation rules so future refactors of build_cpcv_folds / _build_1_factorization
cannot drift silently.

Published rules (LdP AFML Ch.12):
  - K=10 groups, n_test=2 → C(10,2) = 45 total (train+test) combinations.
  - Each combo uses 8 training groups + 2 test groups.
  - Purge: drop training observations within purge_bars of each test boundary.
  - Embargo: drop training observations immediately after each test fold end.
  - 1-factorization (round-robin): partitions all 45 combos into 9 perfect
    matchings, each covering all 10 folds exactly once (for stitching paths).

Source: López de Prado, M. (2018). "Advances in Financial Machine Learning".
Wiley. Chapter 12 (Combinatorial Purged Cross-Validation).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s3_validation.cpcv import (
    build_cpcv_folds,
    _build_1_factorization,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def long_index() -> pd.DatetimeIndex:
    """~3000 business days — large enough for 10-fold CPCV with purge+embargo."""
    return pd.bdate_range(start="2010-01-04", periods=3000)


@pytest.fixture
def short_index() -> pd.DatetimeIndex:
    """~500 business days — marginal; may still produce folds if chunk ≥ 20."""
    return pd.bdate_range(start="2020-01-02", periods=500)


# ---------------------------------------------------------------------------
# C(K, n_test) combination count — LdP AFML Ch.12
# ---------------------------------------------------------------------------

class TestCombinationCount:
    """LdP Ch.12: K=10, n_test=2 → C(10,2) = 45 combinations."""

    def test_c_10_2_equals_45(self) -> None:
        """Published value: C(10,2) = 45 total test-fold combos.

        LdP AFML Ch.12 (p. 200, Table 12.1).
        """
        # Independently verify the math
        expected_combos = len(list(combinations(range(10), 2)))
        assert expected_combos == 45, "C(10,2) must equal 45"

    def test_build_cpcv_folds_returns_at_most_45(
        self, long_index: pd.DatetimeIndex,
    ) -> None:
        """build_cpcv_folds with K=10, n_test=2 → ≤ 45 folds (may drop tiny folds)."""
        folds = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=0, embargo_bars=0,
        )
        # Published max = 45 (some may be dropped if IS too small after purge)
        assert len(folds) <= 45, f"Expected ≤ 45 folds, got {len(folds)}"

    def test_build_cpcv_folds_without_purge_equals_45(
        self, long_index: pd.DatetimeIndex,
    ) -> None:
        """Without purge/embargo, all C(10,2)=45 combos should survive.

        LdP Ch.12: every combo is valid when the series is large enough and
        no observations are excluded.
        """
        folds = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=0, embargo_bars=0,
        )
        assert len(folds) == 45, (
            f"Expected 45 folds (C(10,2)), got {len(folds)}"
        )

    def test_each_combo_has_8_training_groups(
        self, long_index: pd.DatetimeIndex,
    ) -> None:
        """LdP Ch.12: K=10, n_test=2 → 8 training groups per combo.

        We verify this via IS size: with purge_bars=0 and a regular partition,
        IS bars ≈ 8/10 of total bars.
        """
        n = len(long_index)
        chunk = n // 10  # per-fold size
        n_training_groups = 8  # K − n_test = 10 − 2

        folds = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=0, embargo_bars=0,
        )
        # Every IS set should have approximately 8 chunks of training data
        expected_is_bars = n_training_groups * chunk
        for k, (train_pos, _) in enumerate(folds):
            is_bars = len(train_pos)
            # Allow ±1 chunk tolerance for boundary effects
            assert abs(is_bars - expected_is_bars) <= chunk, (
                f"Combo {k}: IS={is_bars} bars, expected ~{expected_is_bars} "
                f"(8 training groups × chunk={chunk})"
            )


# ---------------------------------------------------------------------------
# Purge + embargo reduce IS size — LdP Ch.12 §12.2
# ---------------------------------------------------------------------------

class TestPurgeAndEmbargo:
    """Purge and embargo must shrink IS relative to unpurged baseline."""

    def test_purge_reduces_is_size(self, long_index: pd.DatetimeIndex) -> None:
        """Purge removes observations near test boundaries → IS shrinks.

        LdP Ch.12 §12.2: avoid leakage from overlapping label windows.
        Published effect: IS set loses purge_bars bars on each side of test folds.
        """
        folds_no_purge = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=0, embargo_bars=0,
        )
        folds_purged = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=20, embargo_bars=0,
        )
        # Average IS size must be strictly smaller with purge
        if folds_no_purge and folds_purged:
            avg_is_no_purge = np.mean([len(tr) for tr, _ in folds_no_purge])
            avg_is_purged = np.mean([len(tr) for tr, _ in folds_purged])
            assert avg_is_purged < avg_is_no_purge, (
                f"Purge should shrink IS: no_purge={avg_is_no_purge:.0f}, "
                f"purged={avg_is_purged:.0f}"
            )

    def test_embargo_further_reduces_is(self, long_index: pd.DatetimeIndex) -> None:
        """Embargo additionally removes IS bars after test fold ends → IS shrinks further."""
        folds_purge_only = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=20, embargo_bars=0,
        )
        folds_purge_and_embargo = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=20, embargo_bars=5,
        )
        if folds_purge_only and folds_purge_and_embargo:
            avg_is_purge = np.mean([len(tr) for tr, _ in folds_purge_only])
            avg_is_both = np.mean([len(tr) for tr, _ in folds_purge_and_embargo])
            assert avg_is_both <= avg_is_purge, (
                "Adding embargo should not increase IS size"
            )

    def test_train_and_test_never_overlap(
        self, long_index: pd.DatetimeIndex,
    ) -> None:
        """Critical CPCV invariant: train and test position sets must not overlap."""
        folds = build_cpcv_folds(
            long_index, n_folds=10, n_test_folds=2,
            purge_bars=20, embargo_bars=5,
        )
        for k, (train_pos, test_pos) in enumerate(folds):
            train_set = set(train_pos.tolist())
            test_set = set(test_pos.tolist())
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Fold {k}: train/test overlap = {len(overlap)} positions"
            )

    def test_empty_for_too_short_series(self) -> None:
        """Series too short for 10-fold chunk ≥ 20 → empty list (not crash)."""
        tiny = pd.bdate_range(start="2020-01-01", periods=100)
        folds = build_cpcv_folds(tiny, n_folds=10, n_test_folds=2)
        assert folds == []


# ---------------------------------------------------------------------------
# 1-factorization — round-robin path construction
# ---------------------------------------------------------------------------

class TestOnefactorization:
    """LdP AFML Ch.12: 1-factorization of K_{10} produces 9 perfect matchings.

    Each matching covers all 10 folds exactly once (5 pairs of 2 = 10 folds).
    Together the 9 matchings partition all C(10,2) = 45 edges.
    """

    def test_produces_9_matchings_for_k10(self) -> None:
        """K_{10}: n_folds − 1 = 9 matchings."""
        matchings = _build_1_factorization(10)
        # Published: K_{n} has n−1 perfect matchings (round-robin tournament)
        assert len(matchings) == 9, (
            f"Expected 9 matchings for K_10, got {len(matchings)}"
        )

    def test_each_matching_covers_5_pairs(self) -> None:
        """Each matching is a perfect matching of 10 vertices → 5 pairs."""
        matchings = _build_1_factorization(10)
        for i, matching in enumerate(matchings):
            assert len(matching) == 5, (
                f"Matching {i}: {len(matching)} pairs, expected 5 "
                "(perfect matching of 10 vertices)"
            )

    def test_each_matching_covers_all_10_folds(self) -> None:
        """Each matching covers all 10 fold indices exactly once."""
        matchings = _build_1_factorization(10)
        for i, matching in enumerate(matchings):
            folds_in_matching = [fold for pair in matching for fold in pair]
            assert sorted(folds_in_matching) == list(range(10)), (
                f"Matching {i} does not cover all 10 folds: {folds_in_matching}"
            )

    def test_all_45_edges_covered_exactly_once(self) -> None:
        """All 9 matchings together partition C(10,2)=45 edges exactly once.

        LdP Ch.12: every (i, j) pair appears in exactly one matching.
        """
        matchings = _build_1_factorization(10)
        edge_counts: dict[tuple[int, int], int] = {}
        for matching in matchings:
            for pair in matching:
                a, b = sorted(pair)
                edge: tuple[int, int] = (a, b)
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Exactly 45 distinct edges
        assert len(edge_counts) == 45, (
            f"Expected 45 distinct edges, got {len(edge_counts)}"
        )
        # Each edge covered exactly once
        multi_covered = {e: c for e, c in edge_counts.items() if c != 1}
        assert not multi_covered, (
            f"Edges covered more than once: {multi_covered}"
        )

    def test_folds_within_each_pair_are_distinct(self) -> None:
        """No self-loop: each pair (i, j) must have i ≠ j."""
        matchings = _build_1_factorization(10)
        for i, matching in enumerate(matchings):
            for pair in matching:
                assert pair[0] != pair[1], (
                    f"Matching {i}: self-loop at fold {pair[0]}"
                )
