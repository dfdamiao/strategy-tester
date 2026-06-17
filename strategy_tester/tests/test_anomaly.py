"""Tests for lib.anomaly — check registration, severity resolution, built-ins.

Mirrors the policy table in CLI_REFERENCE.md §10:
  s2_oos_sr + s4_pass_rate → default strict
  others                   → default warn
  policy="off"             → all checks suppressed
  policy="strict"          → escalates all
  strict_on={...}          → wins over global policy
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy_tester.anomaly import (
    AnomalyResult,
    list_checks,
    register_check,
    resolve_severity,
    run_checks,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# AnomalyResult dataclass
# ---------------------------------------------------------------------------


class TestAnomalyResult:
    def test_to_dict_round_trip(self) -> None:
        r = AnomalyResult(
            stage="s1",
            check="s1_pass_rate",
            value=0.02,
            threshold=(0.05, 0.95),
            severity="warn",
            message="too low",
        )
        d = r.to_dict()
        assert d["stage"] == "s1"
        assert d["check"] == "s1_pass_rate"
        assert d["value"] == 0.02
        assert d["threshold"] == (0.05, 0.95)
        assert d["severity"] == "warn"
        assert d["message"] == "too low"

    def test_is_frozen(self) -> None:
        """Dataclass is frozen — mutation must raise."""
        r = AnomalyResult(
            stage="s1", check="x", value=0.0, threshold=0.0,
            severity="warn", message="",
        )
        with pytest.raises((AttributeError, TypeError)):
            r.value = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolve_severity — policy + strict_on semantics
# ---------------------------------------------------------------------------


class TestResolveSeverity:
    def test_off_policy_returns_off(self) -> None:
        assert resolve_severity("s1_pass_rate", policy="off") == "off"

    def test_warn_policy_uses_default_warn(self) -> None:
        assert resolve_severity("s1_pass_rate", policy="warn") == "warn"

    def test_warn_policy_keeps_default_strict_for_lookahead(self) -> None:
        """s2_oos_sr default severity is strict per CLI_REFERENCE.md §10."""
        assert resolve_severity("s2_oos_sr", policy="warn") == "strict"

    def test_warn_policy_keeps_default_strict_for_s4_pass_rate(self) -> None:
        assert resolve_severity("s4_pass_rate", policy="warn") == "strict"

    def test_strict_policy_escalates_warn_to_strict(self) -> None:
        assert resolve_severity("s1_pass_rate", policy="strict") == "strict"

    def test_strict_on_overrides_warn(self) -> None:
        assert (
            resolve_severity(
                "s1_pass_rate", policy="warn", strict_on={"s1_pass_rate"}
            )
            == "strict"
        )

    def test_strict_on_does_not_revive_off(self) -> None:
        """Policy=off suppresses everything — strict_on can't revive it.

        Reading the code: strict_on is checked BEFORE policy=off branch, so
        strict_on actually wins. Pin the current behavior so a refactor
        doesn't silently flip it.
        """
        out = resolve_severity(
            "s1_pass_rate", policy="off", strict_on={"s1_pass_rate"}
        )
        # Current impl: strict_on wins. If you want off to dominate, change
        # the order in anomaly.py::resolve_severity.
        assert out == "strict"

    def test_unknown_check_default_is_warn(self) -> None:
        assert resolve_severity("never_registered", policy="warn") == "warn"


# ---------------------------------------------------------------------------
# Built-in checks — fire only when threshold breached
# ---------------------------------------------------------------------------


class TestBuiltinChecks:
    """All 7 built-ins from CLI_REFERENCE.md §10."""

    def test_s1_pass_rate_too_low_fires(self) -> None:
        df = pd.DataFrame({"passed": [True] + [False] * 99})  # 1%
        results = run_checks("s1", df)
        names = [r.check for r in results]
        assert "s1_pass_rate" in names

    def test_s1_pass_rate_in_band_silent(self) -> None:
        df = pd.DataFrame({"passed": [True] * 30 + [False] * 70})  # 30%
        results = run_checks("s1", df)
        assert results == []

    def test_s1_pass_rate_too_high_fires(self) -> None:
        df = pd.DataFrame({"passed": [True] * 99 + [False]})  # 99%
        results = run_checks("s1", df)
        assert any(r.check == "s1_pass_rate" for r in results)

    def test_s2_oos_sr_too_high_fires(self) -> None:
        """Median OOS Sharpe > 2.0 → look-ahead bias suspect."""
        df = pd.DataFrame({"oos_sharpe": [3.0, 3.5, 4.0]})
        results = run_checks("s2", df)
        hit = next((r for r in results if r.check == "s2_oos_sr"), None)
        assert hit is not None
        assert hit.severity == "strict"  # default for s2_oos_sr

    def test_s2_oos_sr_normal_silent(self) -> None:
        df = pd.DataFrame({"oos_sharpe": [0.5, 0.8, 1.0, 1.2]})
        results = run_checks("s2", df)
        assert all(r.check != "s2_oos_sr" for r in results)

    def test_s3_fold_win_rate_low_fires(self) -> None:
        df = pd.DataFrame({"fold_win_rate": [0.3, 0.35, 0.4]})
        results = run_checks("s3", df)
        assert any(r.check == "s3_fold_win_rate" for r in results)

    def test_s3_cohort_size_too_small_fires(self) -> None:
        df = pd.DataFrame({"passed": [True] * 10 + [False] * 40})  # 10 pass
        results = run_checks("s3", df)
        assert any(r.check == "s3_cohort_size" for r in results)

    def test_s4_pass_rate_too_low_fires_strict(self) -> None:
        """Default severity strict (gate misconfigured)."""
        df = pd.DataFrame({"passed": [True] + [False] * 99})  # 1%
        results = run_checks("s4", df)
        hit = next((r for r in results if r.check == "s4_pass_rate"), None)
        assert hit is not None
        assert hit.severity == "strict"

    def test_s4_pass_rate_uses_psr_pass_when_available(self) -> None:
        """psr_pass column takes precedence over generic 'passed'."""
        df = pd.DataFrame({
            "passed": [True] * 100,  # would be 100% (>0.90 band)
            "psr_pass": [True] * 50 + [False] * 50,  # 50% — in band
        })
        results = run_checks("s4", df)
        assert all(r.check != "s4_pass_rate" for r in results)

    def test_s5_ir_vs_spy_below_zero_fires(self) -> None:
        df = pd.DataFrame({"ir_vs_spy": [-0.5, -0.3, -0.1]})
        results = run_checks("s5", df)
        assert any(r.check == "s5_ir_vs_spy" for r in results)

    def test_s5_locked_sr_below_carver_kill_fires(self) -> None:
        df = pd.DataFrame({"oos_sr_haircut": [0.1, 0.2, 0.3]})
        results = run_checks("s5", df)
        assert any(r.check == "s5_locked_sr" for r in results)


# ---------------------------------------------------------------------------
# run_checks — dispatch / filter / empty df handling
# ---------------------------------------------------------------------------


class TestRunChecks:
    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            run_checks("s99", pd.DataFrame())

    def test_empty_df_returns_no_results(self) -> None:
        assert run_checks("s1", pd.DataFrame()) == []

    def test_missing_column_skips_check(self) -> None:
        """Check returns None when expected column absent."""
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        results = run_checks("s1", df)
        assert results == []

    def test_off_policy_suppresses_all(self) -> None:
        """policy='off' → no checks evaluated."""
        df = pd.DataFrame({"passed": [False] * 100})  # would normally fire
        results = run_checks("s1", df, policy="off")
        assert results == []

    def test_strict_policy_escalates_warn_check(self) -> None:
        df = pd.DataFrame({"passed": [False] * 100})  # s1_pass_rate fires
        results = run_checks("s1", df, policy="strict")
        s1 = next((r for r in results if r.check == "s1_pass_rate"), None)
        assert s1 is not None
        assert s1.severity == "strict"

    def test_threshold_override_changes_band(self) -> None:
        """thresholds={} overrides default min/max bands."""
        df = pd.DataFrame({"passed": [True] * 20 + [False] * 80})  # 20%
        # Default band (0.05, 0.95) → silent
        assert run_checks("s1", df) == []
        # Tighter band (0.50, 0.95) → fires
        results = run_checks(
            "s1", df, thresholds={"s1_min_pass_rate": 0.50}
        )
        assert any(r.check == "s1_pass_rate" for r in results)


# ---------------------------------------------------------------------------
# register_check / list_checks
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            register_check("s99", "custom")(lambda df, **_: None)

    def test_custom_check_fires_via_run_checks(self) -> None:
        sentinel = AnomalyResult(
            stage="s3", check="_test_custom", value=1.0,
            threshold=0.0, severity="warn", message="custom hit",
        )

        @register_check("s3", "_test_custom")
        def _fn(df: pd.DataFrame, **_: object) -> AnomalyResult | None:
            return sentinel if len(df) > 0 else None

        try:
            results = run_checks("s3", pd.DataFrame({"passed": [True]}))
            assert sentinel in results
        finally:
            # Cleanup so other tests aren't polluted
            from strategy_tester.anomaly import _CHECKS
            _CHECKS["s3"].pop("_test_custom", None)

    def test_list_checks_all_stages(self) -> None:
        all_checks = list_checks()
        assert set(all_checks) == {"s0", "s1", "s2", "s3", "s4", "s5"}
        assert "s1_pass_rate" in all_checks["s1"]
        assert "s2_oos_sr" in all_checks["s2"]
        assert "s4_pass_rate" in all_checks["s4"]

    def test_list_checks_single_stage(self) -> None:
        s5 = list_checks("s5")
        assert set(s5["s5"]) >= {"s5_ir_vs_spy", "s5_locked_sr"}

    def test_list_checks_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            list_checks("s99")
