"""Anomaly detection — sanity gates run after each stage.

Built-in checks reproduce the LIB_REBUILD_PLAN.md §5.4 verification
checklist. Policy controls behaviour:

  off     — log only
  warn    — log + emit AnomalyResult (default)
  strict  — log + emit + exit code 2 (caller's job to honour)

Per-check escalation via ``strict_on={check_name, ...}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import pandas as pd

Severity = Literal["off", "warn", "strict"]


@dataclass(frozen=True)
class AnomalyResult:
    """One anomaly observation."""

    stage: str
    check: str
    value: float
    threshold: float | tuple[float, float]
    severity: Severity
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "check": self.check,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
        }


# Check function signature: (df, **thresholds) → AnomalyResult | None
CheckFn = Callable[..., "AnomalyResult | None"]

_CHECKS: dict[str, dict[str, CheckFn]] = {
    "s0": {},
    "s1": {},
    "s2": {},
    "s3": {},
    "s4": {},
    "s5": {},
}

# Per-check default severity from CLI_REFERENCE.md §10
_DEFAULT_SEVERITY: dict[str, Severity] = {
    "s0_min_nonnan": "warn",
    "s1_pass_rate": "warn",
    "s2_oos_sr": "strict",  # look-ahead bias
    "s3_fold_win_rate": "warn",
    "s3_cohort_size": "warn",
    "s4_pass_rate": "strict",  # gate misconfigured
    "s5_ir_vs_spy": "warn",
    "s5_locked_sr": "warn",
}


def register_check(stage: str, name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator to register an anomaly check for a stage."""
    if stage not in _CHECKS:
        raise ValueError(f"Unknown stage {stage!r}")

    def decorator(fn: CheckFn) -> CheckFn:
        _CHECKS[stage][name] = fn
        return fn

    return decorator


def resolve_severity(
    check_name: str,
    *,
    policy: Severity = "warn",
    strict_on: Iterable[str] = (),
) -> Severity:
    """Resolve effective severity for a check.

    Per-check ``strict_on`` always wins over global ``policy``.
    Default severity from ``_DEFAULT_SEVERITY`` only applies when
    ``policy != "off"``.
    """
    if check_name in set(strict_on):
        return "strict"
    if policy == "off":
        return "off"
    default = _DEFAULT_SEVERITY.get(check_name, "warn")
    # `strict` policy escalates everything except `off` overrides.
    if policy == "strict":
        return "strict"
    return default


def run_checks(
    stage: str,
    df: pd.DataFrame,
    *,
    policy: Severity = "warn",
    strict_on: Iterable[str] = (),
    thresholds: dict[str, object] | None = None,
) -> list[AnomalyResult]:
    """Run all registered checks for ``stage`` against ``df``."""
    if stage not in _CHECKS:
        raise ValueError(f"Unknown stage {stage!r}")
    results: list[AnomalyResult] = []
    thresholds = thresholds or {}
    strict_set = set(strict_on)

    for check_name, fn in _CHECKS[stage].items():
        severity = resolve_severity(
            check_name, policy=policy, strict_on=strict_set
        )
        if severity == "off":
            continue
        result = fn(df, severity=severity, **thresholds)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Built-in checks (CLI_REFERENCE.md §10 table)
# ---------------------------------------------------------------------------


@register_check("s1", "s1_pass_rate")
def _s1_pass_rate(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s1_min_pass_rate: float = 0.05,
    s1_max_pass_rate: float = 0.95,
    **_: object,
) -> AnomalyResult | None:
    if "passed" not in df.columns or len(df) == 0:
        return None
    rate = float(df["passed"].mean())
    if rate < s1_min_pass_rate or rate > s1_max_pass_rate:
        return AnomalyResult(
            stage="s1",
            check="s1_pass_rate",
            value=rate,
            threshold=(s1_min_pass_rate, s1_max_pass_rate),
            severity=severity,
            message=(
                f"S1 pass rate {rate:.1%} outside "
                f"[{s1_min_pass_rate:.0%}, {s1_max_pass_rate:.0%}] — "
                "universe filter may be miscalibrated"
            ),
        )
    return None


@register_check("s2", "s2_oos_sr")
def _s2_oos_sr(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s2_oos_sr_max: float = 2.0,
    **_: object,
) -> AnomalyResult | None:
    if "oos_sharpe" not in df.columns or len(df) == 0:
        return None
    median = float(df["oos_sharpe"].median())
    if median > s2_oos_sr_max:
        return AnomalyResult(
            stage="s2",
            check="s2_oos_sr",
            value=median,
            threshold=s2_oos_sr_max,
            severity=severity,
            message=(
                f"S2 median OOS Sharpe {median:.2f} > {s2_oos_sr_max:.2f} — "
                "likely look-ahead bias"
            ),
        )
    return None


@register_check("s3", "s3_fold_win_rate")
def _s3_fold_win_rate(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s3_fold_win_min: float = 0.5,
    **_: object,
) -> AnomalyResult | None:
    col = "fold_win_rate" if "fold_win_rate" in df.columns else None
    if col is None or len(df) == 0:
        return None
    median = float(df[col].median())
    if median < s3_fold_win_min:
        return AnomalyResult(
            stage="s3",
            check="s3_fold_win_rate",
            value=median,
            threshold=s3_fold_win_min,
            severity=severity,
            message=(
                f"S3 median fold_win_rate {median:.2f} < "
                f"{s3_fold_win_min:.2f} — S2 likely overfit"
            ),
        )
    return None


@register_check("s3", "s3_cohort_size")
def _s3_cohort_size(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s3_cohort_min: int = 50,
    **_: object,
) -> AnomalyResult | None:
    if "passed" not in df.columns:
        return None
    n_passed = int(df["passed"].sum())
    if n_passed < s3_cohort_min:
        return AnomalyResult(
            stage="s3",
            check="s3_cohort_size",
            value=n_passed,
            threshold=s3_cohort_min,
            severity=severity,
            message=(
                f"S3 passing cohort {n_passed} < {s3_cohort_min} — "
                "may not survive S4 multiple-testing"
            ),
        )
    return None


@register_check("s4", "s4_pass_rate")
def _s4_pass_rate(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s4_pass_rate_min: float = 0.05,
    s4_pass_rate_max: float = 0.90,
    **_: object,
) -> AnomalyResult | None:
    col = "psr_pass" if "psr_pass" in df.columns else "passed"
    if col not in df.columns or len(df) == 0:
        return None
    rate = float(df[col].mean())
    if rate < s4_pass_rate_min or rate > s4_pass_rate_max:
        return AnomalyResult(
            stage="s4",
            check="s4_pass_rate",
            value=rate,
            threshold=(s4_pass_rate_min, s4_pass_rate_max),
            severity=severity,
            message=(
                f"S4 pass rate {rate:.1%} outside "
                f"[{s4_pass_rate_min:.0%}, {s4_pass_rate_max:.0%}] — "
                "gate misconfigured"
            ),
        )
    return None


@register_check("s5", "s5_ir_vs_spy")
def _s5_ir_vs_spy(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s5_ir_min: float = 0.0,
    **_: object,
) -> AnomalyResult | None:
    if "ir_vs_spy" not in df.columns or len(df) == 0:
        return None
    top = float(df["ir_vs_spy"].max())
    if top < s5_ir_min:
        return AnomalyResult(
            stage="s5",
            check="s5_ir_vs_spy",
            value=top,
            threshold=s5_ir_min,
            severity=severity,
            message=(
                f"S5 top scheme IR vs SPY {top:.3f} < {s5_ir_min:.3f} — "
                "no edge over benchmark"
            ),
        )
    return None


@register_check("s5", "s5_locked_sr")
def _s5_locked_sr(
    df: pd.DataFrame,
    *,
    severity: Severity = "warn",
    s5_locked_sr_min: float = 0.5,
    **_: object,
) -> AnomalyResult | None:
    col = "oos_sr_haircut" if "oos_sr_haircut" in df.columns else None
    if col is None or len(df) == 0:
        return None
    top = float(df[col].max())
    if top < s5_locked_sr_min:
        return AnomalyResult(
            stage="s5",
            check="s5_locked_sr",
            value=top,
            threshold=s5_locked_sr_min,
            severity=severity,
            message=(
                f"S5 top DSR-adjusted SR {top:.2f} < {s5_locked_sr_min:.2f} — "
                "Carver kill threshold"
            ),
        )
    return None


def list_checks(stage: str | None = None) -> dict[str, list[str]]:
    """List registered anomaly checks (for CLI --help)."""
    if stage is not None:
        if stage not in _CHECKS:
            raise ValueError(f"Unknown stage {stage!r}")
        return {stage: sorted(_CHECKS[stage])}
    return {s: sorted(c) for s, c in _CHECKS.items()}
