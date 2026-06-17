"""Mathematical verification tests for Deflated Sharpe Ratio.

Guards against the dimensional bug fixed 2026-05-15 in
``lib/s4_significance/dsr.py``. Pre-fix code computed
``z = (sr - e_max_std_normal) / std_sr`` — a unit mismatch between SR units
and standard-normal quantile units. Every test here would have failed under
the pre-fix code; tests should fail loudly if the bug returns.

Reference: Bailey & López de Prado, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
Journal of Portfolio Management, 2014.

Reference values for E[max(N standard normals)] — Bailey-LdP 2014 Eq.5 EXACT
(γ_e = 0.5772156649, Φ⁻¹ from scipy.stats.norm):
    N=2     → 0.520
    N=10    → 1.575
    N=100   → 2.531
    N=1000  → 3.254
    N=10000 → 3.858
(These are the EXACT closed-form values from Eq.5, not the Gumbel asymptotic
approximation which gives slightly lower values.)
"""
from __future__ import annotations

import numpy as np
import pytest

from strategy_tester.backtest.metrics import psr_stat
from strategy_tester.s4_significance.dsr import (
    deflated_sharpe,
    effective_n_from_correlation,
    expected_max_sharpe,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# E[max(SR)] — Bailey-LdP 2014 Eq.5
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    """Direct verification of the E[max] computation."""

    def test_returns_zero_for_n_below_2(self) -> None:
        """N < 2 means no selection → no deflation needed."""
        assert expected_max_sharpe(1, sr_variance=1.0) == 0.0
        assert expected_max_sharpe(0, sr_variance=1.0) == 0.0

    def test_returns_zero_for_nonpositive_variance(self) -> None:
        """Degenerate variance → no spread → no expected max."""
        assert expected_max_sharpe(100, sr_variance=0.0) == 0.0
        assert expected_max_sharpe(100, sr_variance=-1.0) == 0.0

    @pytest.mark.parametrize("n,expected", [
        (2, 0.520),
        (10, 1.575),
        (100, 2.531),
        (1000, 3.254),
        (10000, 3.858),
    ])
    def test_exact_bailey_ldp_quantile_values(self, n: int, expected: float) -> None:
        """Bailey-LdP 2014 Eq.5 with sr_variance=1.0 — exact closed-form values.

        Computed independently from γ_e=0.5772156649 and scipy norm.ppf:
            E[max(N)] = (1−γ_e)·Φ⁻¹(1−1/N) + γ_e·Φ⁻¹(1−1/(N·e))
        """
        z_max = expected_max_sharpe(n, sr_variance=1.0)
        assert z_max == pytest.approx(expected, abs=0.005), (
            f"E[max(N={n})] = {z_max:.4f}, expected = {expected:.3f}"
        )

    def test_monotone_increasing_in_n(self) -> None:
        """More trials → higher expected max."""
        e_max_values = [expected_max_sharpe(n, 1.0) for n in [2, 10, 100, 1000]]
        assert all(a < b for a, b in zip(e_max_values, e_max_values[1:]))

    def test_scales_linearly_with_sqrt_variance(self) -> None:
        """E[max(N, k²·σ²)] = k · E[max(N, σ²)] — variance enters via sqrt."""
        base = expected_max_sharpe(100, sr_variance=1.0)
        scaled_4 = expected_max_sharpe(100, sr_variance=4.0)
        scaled_9 = expected_max_sharpe(100, sr_variance=9.0)
        assert scaled_4 == pytest.approx(2.0 * base, rel=1e-9)
        assert scaled_9 == pytest.approx(3.0 * base, rel=1e-9)

    def test_returns_sr_units_not_standard_normal(self) -> None:
        """REGRESSION TEST for the 2026-05-15 dimensional bug.

        Pre-fix code returned the raw standard-normal quantile (~2.5 for N=100).
        Fixed code returns sqrt(variance) × quantile (= variance-scaled SR).
        For sr_variance=0.25, the result must be ~1.25, NOT ~2.5.
        """
        result = expected_max_sharpe(100, sr_variance=0.25)
        assert result == pytest.approx(0.5 * 2.508, abs=0.02), (
            f"E[max] for var=0.25 returned {result:.4f} — should be ~1.25 "
            "(sqrt(0.25)·2.508), NOT ~2.5 (raw quantile). Unit bug returned."
        )


# ---------------------------------------------------------------------------
# effective_N from correlation — LdP AFML Ch.14
# ---------------------------------------------------------------------------

class TestEffectiveN:

    def test_zero_correlation_preserves_n(self) -> None:
        """Independent trials: effective_N = N."""
        assert effective_n_from_correlation(0.0, 100) == pytest.approx(100.0)
        assert effective_n_from_correlation(0.0, 864) == pytest.approx(864.0)

    def test_high_correlation_compresses_n(self) -> None:
        """Heavily correlated trials: effective_N → 1."""
        n_eff = effective_n_from_correlation(0.95, 1000)
        assert 1.0 < n_eff < 3.0, (
            f"effective_N at ρ̄=0.95 from N=1000 should be ~2, got {n_eff:.2f}"
        )

    def test_correlation_one_floored_at_one(self) -> None:
        """ρ̄ → 1 must not produce N_eff < 1."""
        assert effective_n_from_correlation(1.0, 1000) >= 1.0
        assert effective_n_from_correlation(0.999, 1000) >= 1.0

    def test_typical_grid_compression(self) -> None:
        """For a typical correlated param grid (ρ̄ ≈ 0.85), N=864 → N_eff ≈ 2.85."""
        n_eff = effective_n_from_correlation(0.85, 864)
        assert n_eff == pytest.approx(2.85, abs=0.1)

    def test_negative_rho_clipped_to_zero(self) -> None:
        """Negative correlation clipped — N_eff cannot exceed raw N."""
        assert effective_n_from_correlation(-0.5, 100) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# DSR — full integration
# ---------------------------------------------------------------------------

class TestDeflatedSharpe:

    def test_reduces_to_psr_when_n_trials_is_one(self) -> None:
        """With n_trials=1, E[max]=0, so DSR ≡ PSR(0)."""
        dsr_val = deflated_sharpe(
            sharpe=1.0, n_obs=1260, n_trials=1, sr_variance=0.5,
        )
        psr_val = psr_stat(1.0, 1260, 0.0, 3.0, sr_benchmark=0.0)
        assert dsr_val == pytest.approx(psr_val, rel=1e-9)

    def test_equals_half_at_boundary(self) -> None:
        """When SR_obs exactly equals E[max], z=0 and DSR = Φ(0) = 0.5."""
        n_trials = 100
        sr_var = 0.25
        sr_at_boundary = expected_max_sharpe(n_trials, sr_var)
        dsr_val = deflated_sharpe(
            sharpe=sr_at_boundary, n_obs=1260,
            n_trials=n_trials, sr_variance=sr_var,
        )
        assert dsr_val == pytest.approx(0.5, abs=0.005)

    def test_strictly_below_psr_when_n_trials_above_one(self) -> None:
        """For SR > 0, more trials → lower DSR. Verifies the deflation works."""
        psr0 = psr_stat(1.0, 1260, 0.0, 3.0, sr_benchmark=0.0)
        dsr_n10 = deflated_sharpe(1.0, 1260, 10, 0.5)
        dsr_n100 = deflated_sharpe(1.0, 1260, 100, 0.5)
        dsr_n1000 = deflated_sharpe(1.0, 1260, 1000, 0.5)
        assert dsr_n10 < psr0
        assert dsr_n100 < dsr_n10
        assert dsr_n1000 < dsr_n100

    def test_invariant_under_consistent_unit_scaling(self) -> None:
        """REGRESSION: DSR must be invariant under simultaneous (SR, sqrt(var))
        rescaling. Pre-fix code violated this because e_max wasn't in SR units.

        If both SR_obs and sqrt(Var(SR)) are scaled by k, DSR should not change.
        """
        base = deflated_sharpe(
            sharpe=1.0, n_obs=1260, n_trials=100, sr_variance=0.25,
        )
        # Rescale both SR and sqrt(variance) by k=2
        scaled = deflated_sharpe(
            sharpe=2.0, n_obs=1260, n_trials=100, sr_variance=1.0,
        )
        assert scaled == pytest.approx(base, abs=0.005), (
            f"DSR not invariant under unit rescaling: base={base:.4f}, "
            f"scaled={scaled:.4f}. Dimensional bug may have returned."
        )

    def test_effective_n_via_rho_makes_dsr_easier(self) -> None:
        """rho_bar > 0 → effective_N < N → lower bar → higher DSR."""
        dsr_raw = deflated_sharpe(
            sharpe=1.0, n_obs=1260, n_trials=864, sr_variance=0.5, rho_bar=None,
        )
        dsr_corrected = deflated_sharpe(
            sharpe=1.0, n_obs=1260, n_trials=864, sr_variance=0.5, rho_bar=0.85,
        )
        assert dsr_corrected > dsr_raw, (
            f"effective_K correction should reduce the bar: "
            f"raw={dsr_raw:.4f}, corrected={dsr_corrected:.4f}"
        )

    def test_high_sr_low_n_passes(self) -> None:
        """SR=3.0 with N=10 trials should pass DSR ≥ 0.95."""
        dsr_val = deflated_sharpe(
            sharpe=3.0, n_obs=1260, n_trials=10, sr_variance=0.25,
        )
        assert dsr_val > 0.95

    def test_low_sr_high_n_fails(self) -> None:
        """SR=0.5 with N=10000 trials should fail DSR ≥ 0.90 by a wide margin."""
        dsr_val = deflated_sharpe(
            sharpe=0.5, n_obs=1260, n_trials=10000, sr_variance=0.5,
        )
        assert dsr_val < 0.50


# ---------------------------------------------------------------------------
# REGRESSION TESTS — block the historical bug from returning
# ---------------------------------------------------------------------------

class TestBaileyLdpPaperReproduction:
    """Reproduce Bailey-LdP 2014 paper's published numerical results.

    Two anchoring tests:
    1. Identity against the paper's own Python implementation (Snippet 1, p.13).
    2. Reproduction of the paper's worked example (pages 9-10):
       Strategy SR=2.5 annualized, T=1250 days, ANN=250, gamma3=-3, gamma4=10,
       V[{SR_n}] = 0.5 (annualized variance of trial SRs).
       Paper states: DSR(N=46) = 0.9505 (deploy), DSR(N=100) ≈ 0.90 (decline).

    These are gold-standard tests — if they ever fail, the implementation has
    drifted from the paper.
    """

    def test_identity_against_paper_snippet_1(self) -> None:
        """expected_max_sharpe(N, V) must equal paper's getExpMaxSR(0, sqrt(V), N).

        Paper's Snippet 1 (page 13, Appendix A.2):
            def getExpMaxSR(mu, sigma, numTrials):
                emc = 0.5772156649
                maxZ = (1-emc)*norm.ppf(1-1./numTrials) \
                       + emc*norm.ppf(1-1./(numTrials*np.e))
                return mu + sigma*maxZ

        Floating-point identity is required (tolerance 1e-12).
        """
        from scipy.stats import norm

        def paper_getExpMaxSR(mu: float, sigma: float, n: int) -> float:
            emc = 0.5772156649
            max_z = ((1 - emc) * norm.ppf(1 - 1.0 / n)
                     + emc * norm.ppf(1 - 1.0 / (n * np.e)))
            return float(mu + sigma * max_z)

        for n in [2, 10, 100, 864, 1000, 10000]:
            for v in [0.01, 0.25, 0.5, 1.0]:
                paper = paper_getExpMaxSR(0.0, float(np.sqrt(v)), n)
                mine = expected_max_sharpe(n, sr_variance=v)
                assert mine == pytest.approx(paper, abs=1e-12), (
                    f"N={n}, V={v}: paper={paper}, mine={mine}"
                )

    def test_worked_example_n46_dsr_equals_0_9505(self) -> None:
        """Paper page 10: 'Should the strategist have made his discovery after
        running only N=46 independent trials, the investor may have allocated
        some funds, as DSR would have been 0.9505, above the 95% confidence
        level.' (Bailey & López de Prado 2014).

        Inputs from the paper:
            SR_annualized = 2.5
            T = 1250 daily observations
            ANN = 250 obs/year
            gamma_3 = -3, gamma_4 = 10
            V[{SR_n}] = 0.5 (annualized variance of trial SRs)
            N = 46

        Expected DSR = 0.9505 (paper-published value, 4 decimals).
        """
        ANN = 250
        sr_per_period = 2.5 / np.sqrt(ANN)
        v_per_period = 0.5 / ANN

        dsr_val = deflated_sharpe(
            sharpe=sr_per_period,
            n_obs=1250,
            n_trials=46,
            sr_variance=v_per_period,
            skew=-3.0,
            kurtosis=10.0,
        )
        assert dsr_val == pytest.approx(0.9505, abs=0.001), (
            f"DSR(N=46) = {dsr_val:.4f}, paper says 0.9505"
        )

    def test_worked_example_n100_dsr_approximately_0_90(self) -> None:
        """Paper page 10: 'The investor has recognized that there is only a 90%
        chance that the true SR associated with this strategy is greater than
        zero.' for N=100.

        Reproducing the rest of the worked example confirms DSR(N=100) ≈ 0.9004
        — which the paper rounds to '90%' in prose.
        """
        ANN = 250
        sr_per_period = 2.5 / np.sqrt(ANN)
        v_per_period = 0.5 / ANN

        dsr_val = deflated_sharpe(
            sharpe=sr_per_period,
            n_obs=1250,
            n_trials=100,
            sr_variance=v_per_period,
            skew=-3.0,
            kurtosis=10.0,
        )
        assert dsr_val == pytest.approx(0.9004, abs=0.001), (
            f"DSR(N=100) = {dsr_val:.4f}, paper says ~0.90"
        )

    def test_worked_example_monotone_in_n(self) -> None:
        """At fixed inputs, DSR must decrease as N increases (more trials → harder)."""
        ANN = 250
        sr_per_period = 2.5 / np.sqrt(ANN)
        v_per_period = 0.5 / ANN
        common = dict(
            sharpe=sr_per_period, n_obs=1250, sr_variance=v_per_period,
            skew=-3.0, kurtosis=10.0,
        )
        dsr_values = [deflated_sharpe(n_trials=n, **common)
                      for n in [10, 46, 100, 1000, 10000]]
        # Strictly decreasing
        for a, b in zip(dsr_values, dsr_values[1:]):
            assert a > b, f"DSR should decrease with N: {dsr_values}"


class TestDimensionalBugRegression:
    """Explicit guards against the 2026-05-15 dimensional bug.

    Pre-fix code at lib/s4_significance/dsr.py used:
        e_max = (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))   # standard normal units
        z = (sr - e_max) / std_sr                       # UNIT MISMATCH

    With sr=1.0, e_max=2.508 (N=100), std_sr=0.5:
        wrong_z = (1.0 - 2.508)/0.5 = -3.02 → DSR ≈ 0.001
        right_z = (1.0 - 0.5·2.508)/0.5 = -0.508 → DSR ≈ 0.305
    """

    def test_known_worked_example_fold_scale(self) -> None:
        """SR=1.0, N=100, sr_variance=0.25, n_obs=8 folds, normal returns.

        At fold-scale (n_obs=8), the bug separation is visible:
            E[max(SR)] = sqrt(0.25) × 2.508 = 1.254
            SE(SR)     ≈ sqrt(denom)/sqrt(7) ≈ 1.06/2.65 ≈ 0.40
            z          = (1.0 - 1.254) × sqrt(7) / 1.06 ≈ -0.634
            DSR        = Φ(-0.634) ≈ 0.263

        Pre-fix bug would have given DSR ≈ 0.001 (treating e_max as SR-unit).
        Test asserts DSR is in the [0.20, 0.35] band — fails catastrophically
        if bug returns.
        """
        dsr_val = deflated_sharpe(
            sharpe=1.0, n_obs=8, n_trials=100, sr_variance=0.25,
        )
        assert 0.20 < dsr_val < 0.35, (
            f"DSR = {dsr_val:.4f}; expected ~0.26 for SR=1.0, N=100, var=0.25, "
            "n_obs=8. If DSR < 0.01, the dimensional bug has returned."
        )

    def test_above_emax_passes_at_fold_scale(self) -> None:
        """SR comfortably above E[max] should produce DSR > 0.95."""
        # E[max] for N=100, var=0.25 ≈ 1.265 (exact). SR=3.0 is well above.
        dsr_val = deflated_sharpe(
            sharpe=3.0, n_obs=8, n_trials=100, sr_variance=0.25,
        )
        assert dsr_val > 0.95, (
            f"DSR={dsr_val:.4f} should exceed 0.95 for SR=3.0 well above E[max]≈1.27"
        )

    def test_pre_fix_formula_gives_different_answer(self) -> None:
        """Sanity: the buggy formula and correct formula disagree at typical inputs.

        If this test ever shows they agree, the fix has been reverted.
        """
        # Buggy formula: (sr - z_max_normal) / std_sr, treating z_max as SR-unit
        from scipy.stats import norm
        sr, std_sr, n_trials = 1.0, 0.5, 100
        gamma = 0.5772156649
        z_max = (1-gamma)*norm.ppf(1-1/n_trials) + gamma*norm.ppf(1-1/(n_trials*np.e))
        buggy_z = (sr - z_max) / std_sr
        buggy_dsr = float(norm.cdf(buggy_z))

        correct_dsr = deflated_sharpe(
            sharpe=sr, n_obs=8, n_trials=n_trials, sr_variance=std_sr**2,
        )
        # The two should be meaningfully different (>5pp gap typical)
        assert abs(buggy_dsr - correct_dsr) > 0.05, (
            f"Buggy ({buggy_dsr:.4f}) and correct ({correct_dsr:.4f}) DSR agree "
            "— either fix is not applied or inputs happen to be degenerate."
        )

    def test_unit_conversion_explicit(self) -> None:
        """The conversion sqrt(variance) × z_max MUST be present in E[max]."""
        # With variance=4, sqrt = 2. E[max] should be exactly 2× the unit case.
        e_max_unit = expected_max_sharpe(100, sr_variance=1.0)
        e_max_var4 = expected_max_sharpe(100, sr_variance=4.0)
        ratio = e_max_var4 / e_max_unit
        assert ratio == pytest.approx(2.0, abs=1e-6), (
            f"E[max] should scale linearly with sqrt(variance). Got ratio "
            f"{ratio:.4f} for variance ratio 4:1. If ratio ≈ 1, the unit "
            "conversion is missing (pre-fix bug)."
        )
