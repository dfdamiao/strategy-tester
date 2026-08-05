"""Regression tests for the PSR/DSR frequency fix (ported 2026-08-05).

``n_obs`` = daily bar count (correct, Bailey-LdP), but the Sharpe fed
alongside that daily bar count was ANNUALIZED, inflating the PSR/DSR
z-statistic by ~sqrt(252) and saturating the DSR-95 gate. The fix
de-annualizes SR (and its cross-trial variance) at the S4 gate boundaries
so both match the bar frequency.

These tests assert the boundary now de-annualizes, that a mediocre
annualized SR no longer passes the 95% gate (the bug), and that a genuinely
strong one still does. Plus the E[max] continuity fix at N_eff=2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.backtest.metrics import to_per_period
from strategy_tester.s4_significance.dsr import (
    deflated_sharpe,
    dsr,
    effective_n_from_correlation,
    expected_max_sharpe,
)
from strategy_tester.s4_significance.psr import psr

pytestmark = pytest.mark.unit


def test_to_per_period_deannualizes() -> None:
    sr_pp, var_pp = to_per_period(2.0, 0.30)
    assert sr_pp == pytest.approx(2.0 / np.sqrt(252))
    assert var_pp == pytest.approx(0.30 / 252)
    # variance omitted -> None
    assert to_per_period(1.0)[1] is None


def _s3_row(sr_ann: float, n_obs: int = 1500, sr_var: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "AAA/AAA",
                "numerator": "AAA",
                "denominator": "AAA",
                "passed": True,
                "mean_test_sharpe": sr_ann,
                "std_test_sharpe": np.sqrt(sr_var),
                "sr_variance": sr_var,
                "n_test_periods": 8,
                "n_oos_bars": n_obs,
            }
        ]
    )


class TestGateBoundaryFrequencyConsistent:
    def test_dsr_matches_manual_deannualization(self) -> None:
        """dsr() must equal deflated_sharpe with hand-de-annualized inputs."""
        out = dsr(_s3_row(0.9), n_trials=100, rho_bar=0.70)
        sr_pp, var_pp = to_per_period(0.9, 0.02)
        want = deflated_sharpe(
            sharpe=sr_pp, n_obs=1500, n_trials=100, sr_variance=var_pp, rho_bar=0.70
        )
        assert out.iloc[0]["dsr_stat"] == pytest.approx(want, abs=1e-4)

    def test_mediocre_annualized_sr_no_longer_passes_dsr(self) -> None:
        """The bug: annualized SR 0.5 gave DSR~1.0. Fixed: it must NOT pass."""
        out = dsr(_s3_row(0.5), n_trials=100, rho_bar=0.70)
        assert out.iloc[0]["dsr_stat"] < 0.95
        assert not bool(out.iloc[0]["dsr_passed"])

    def test_strong_annualized_sr_still_passes_dsr(self) -> None:
        """A genuinely strong edge (annualized SR 2.0) must still clear 95%."""
        out = dsr(_s3_row(2.0), n_trials=100, rho_bar=0.70)
        assert out.iloc[0]["dsr_stat"] > 0.95
        assert bool(out.iloc[0]["dsr_passed"])

    def test_mediocre_annualized_sr_no_longer_passes_psr(self) -> None:
        out = psr(_s3_row(0.5))
        assert out.iloc[0]["psr_stat"] < 0.95
        assert not bool(out.iloc[0]["psr_passed"])

    def test_strong_annualized_sr_still_passes_psr(self) -> None:
        out = psr(_s3_row(2.0))
        assert out.iloc[0]["psr_stat"] > 0.95
        assert bool(out.iloc[0]["psr_passed"])


class TestExpectedMaxNoCliff:
    """E[max] must be continuous through N_eff=2, not a cliff."""

    def test_no_discontinuity_at_n_eff_2(self) -> None:
        just_below = expected_max_sharpe(1.99, 0.30)
        at_two = expected_max_sharpe(2.0, 0.30)
        assert just_below == pytest.approx(at_two, rel=0.02)  # was 0.0 vs 0.30

    def test_nonzero_deflation_at_typical_operating_point(self) -> None:
        """rho_bar~0.85, N=96 -> N_eff~1.98 must give real deflation, not 0."""
        n_eff = effective_n_from_correlation(0.85, 96)
        assert n_eff < 2.0  # the regime that used to zero out
        assert expected_max_sharpe(n_eff, 0.30) > 0.0

    def test_monotone_and_zero_at_one(self) -> None:
        assert expected_max_sharpe(1.0, 0.30) == 0.0
        vals = [expected_max_sharpe(n, 0.30) for n in (1.2, 1.5, 1.99, 2.5, 10, 96)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))  # monotone non-decreasing
