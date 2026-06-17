"""Pydantic v2 schemas for stage outputs.

Complements ``interfaces.py`` (frozenset column lists) with row-level type +
range validation. Schemas are the contract `lib.io` enforces when reading or
writing stage artifacts.

Status: Phase 1.E foundation (2026-05-17). Not yet wired into
``lib.pipeline.Pipeline.run`` — that migration is gated on the post-kickoff
parity audit (Phase 1.C).
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__schema_version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Per-row models
# ---------------------------------------------------------------------------


class S1Row(BaseModel):
    """One row of S1 universe-screen output."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    pair: str
    numerator: str
    denominator: str
    passed: bool
    halflife: float | None = None
    window: int | None = Field(default=None, ge=1)
    method: str


class S2Row(BaseModel):
    """One row of S2 signal + optimizer output.

    `signal_method` must propagate downstream — see rules.md §3.
    """

    model_config = ConfigDict(extra="allow")

    pair: str
    numerator: str
    denominator: str
    halflife: float | None = None
    window: int | None = Field(default=None, ge=1)
    entry_thresh: float
    exit_thresh: float
    stop_pct: float = Field(ge=0.0)
    slope_min: float
    is_sharpe: float
    is_penalized_sharpe: float
    is_trades: int = Field(ge=0)
    oos_sharpe: float
    oos_trades: int = Field(ge=0)
    passed: bool
    signal_method: str
    optim_method: str


class S3Row(BaseModel):
    """One row of S3 walk-forward / CV output."""

    model_config = ConfigDict(extra="allow")

    pair: str
    numerator: str
    denominator: str
    mean_test_sharpe: float
    std_test_sharpe: float = Field(ge=0.0)
    n_test_periods: int = Field(ge=1)
    # METHODOLOGY_DECISIONS.md §1: bar count T for paper-correct PSR/DSR.
    # Optional during the legacy → bar-count migration window.
    n_oos_bars: int | None = Field(default=None, ge=0)
    baseline_sharpe: float
    degradation: float
    passed: bool
    val_method: str


class S4Row(BaseModel):
    """One row of S4 significance gate output.

    Schema-locks the columns the S4 → S5 cohort selectors read. Adds
    explicit fields for the `--s4-cohort` CLI choices (passed_s4 /
    passed_s4_strict / psr_pass / psr_pass_strict / dsr_pass / dsr_pass_strict).
    """

    model_config = ConfigDict(extra="allow")

    pair: str
    numerator: str
    denominator: str
    passed: bool
    tier: Literal["TOP_TIER", "SECOND_TIER", "REJECT", "GOOD", "MARGINAL"]

    # Significance metrics (optional — depends on s4.methods config)
    psr_value: float | None = Field(default=None, ge=0.0, le=1.0)
    dsr_value: float | None = Field(default=None, ge=0.0, le=1.0)
    bootstrap_lower_ci: float | None = None

    # Cohort selector flags — match CLI_REFERENCE.md §7
    psr_pass: bool | None = None
    psr_pass_strict: bool | None = None
    dsr_pass: bool | None = None
    dsr_pass_strict: bool | None = None
    passed_s4: bool | None = None
    passed_s4_strict: bool | None = None


class S5Row(BaseModel):
    """One row of S5 per-scheme portfolio output."""

    model_config = ConfigDict(extra="allow")

    scheme_name: str
    cohort: Literal["singles", "ratios", "combined"]
    oos_sr_full: float
    oos_sr_active: float | None = None
    oos_sr_haircut: float | None = None
    oos_cagr: float
    oos_max_dd: float = Field(le=0.0)
    oos_calmar: float | None = None
    ir_vs_spy: float | None = None
    alpha: float | None = None
    beta: float | None = None
    n_units: int = Field(ge=1)


_STAGE_MODELS: dict[str, type[BaseModel]] = {
    "s1": S1Row,
    "s2": S2Row,
    "s3": S3Row,
    "s4": S4Row,
    "s5": S5Row,
}


# ---------------------------------------------------------------------------
# DataFrame validators
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a DataFrame fails schema validation."""


def validate_dataframe(
    df: pd.DataFrame,
    stage: str,
    *,
    strict: bool = False,
    max_errors: int = 5,
) -> pd.DataFrame:
    """Validate every row of ``df`` against the stage schema.

    Parameters
    ----------
    df : DataFrame to validate.
    stage : One of "s1", "s2", "s3", "s4", "s5".
    strict : If True, raise on first row failure. If False (default),
        accumulate up to ``max_errors`` and raise once with all of them.
    max_errors : Max errors to accumulate when ``strict=False``.

    Returns
    -------
    The same DataFrame (unchanged) if validation succeeds.

    Raises
    ------
    SchemaError : If any row fails validation, or stage is unknown.
    """
    if stage not in _STAGE_MODELS:
        raise SchemaError(
            f"Unknown stage {stage!r}. Valid: {sorted(_STAGE_MODELS)}"
        )

    if df.empty:
        return df

    model = _STAGE_MODELS[stage]
    errors: list[str] = []
    records = df.to_dict(orient="records")

    for idx, row in enumerate(records):
        try:
            model.model_validate(row)
        except ValidationError as exc:
            err_msg = f"row {idx}: {exc.errors(include_url=False)}"
            if strict:
                raise SchemaError(f"{stage}: {err_msg}") from exc
            errors.append(err_msg)
            if len(errors) >= max_errors:
                break

    if errors:
        joined = "\n  ".join(errors)
        raise SchemaError(
            f"{stage}: {len(errors)} row(s) failed validation:\n  {joined}"
        )

    return df


def schema_for(stage: str) -> type[BaseModel]:
    """Return the pydantic model class for a stage."""
    if stage not in _STAGE_MODELS:
        raise SchemaError(
            f"Unknown stage {stage!r}. Valid: {sorted(_STAGE_MODELS)}"
        )
    return _STAGE_MODELS[stage]


def list_stages() -> list[str]:
    """List stages that have a registered schema."""
    return sorted(_STAGE_MODELS)
