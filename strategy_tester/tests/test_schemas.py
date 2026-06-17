"""Tests for lib.schemas — pydantic row models + validate_dataframe.

Schemas are complementary to interfaces.py (column-existence) — these add
per-row type and range validation. See lib/CLAUDE.md and schemas.py docstring.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy_tester.schemas import (
    S1Row,
    S2Row,
    S3Row,
    S4Row,
    S5Row,
    SchemaError,
    __schema_version__,
    list_stages,
    schema_for,
    validate_dataframe,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Row models — happy path + boundary fields
# ---------------------------------------------------------------------------


def _s1_row(**overrides: object) -> dict[str, object]:
    base = {
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "passed": True, "halflife": 30.0, "window": 15,
        "method": "chan_halflife",
    }
    base.update(overrides)
    return base


def _s2_row(**overrides: object) -> dict[str, object]:
    base = {
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "halflife": 30.0, "window": 15,
        "entry_thresh": -2.0, "exit_thresh": 0.5, "stop_pct": 0.1,
        "slope_min": 0.0, "is_sharpe": 1.2, "is_penalized_sharpe": 1.0,
        "is_trades": 50, "oos_sharpe": 0.8, "oos_trades": 10,
        "passed": True, "signal_method": "zscore_robust_mad",
        "optim_method": "grid_search",
    }
    base.update(overrides)
    return base


def _s3_row(**overrides: object) -> dict[str, object]:
    base = {
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "mean_test_sharpe": 0.7, "std_test_sharpe": 0.3,
        "n_test_periods": 8, "baseline_sharpe": 1.0,
        "degradation": 0.3, "passed": True, "val_method": "wfa_expanding",
    }
    base.update(overrides)
    return base


def _s4_row(**overrides: object) -> dict[str, object]:
    base = {
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "passed": True, "tier": "GOOD",
    }
    base.update(overrides)
    return base


def _s5_row(**overrides: object) -> dict[str, object]:
    base = {
        "scheme_name": "equal_weight_flat_1k_cap10",
        "cohort": "ratios",
        "oos_sr_full": 1.2, "oos_cagr": 0.18, "oos_max_dd": -0.15,
        "n_units": 30,
    }
    base.update(overrides)
    return base


class TestS1Row:
    def test_valid(self) -> None:
        S1Row.model_validate(_s1_row())

    def test_window_must_be_ge_1(self) -> None:
        with pytest.raises(Exception):  # noqa: B017  ValidationError
            S1Row.model_validate(_s1_row(window=0))

    def test_halflife_nan_allowed(self) -> None:
        """NaN halflife is valid (trend strategies / pass-through)."""
        S1Row.model_validate(_s1_row(halflife=None))

    def test_extra_columns_allowed(self) -> None:
        """extra='allow' — strategy-specific columns must not break."""
        S1Row.model_validate(_s1_row(custom_metric=0.42))


class TestS2Row:
    def test_valid(self) -> None:
        S2Row.model_validate(_s2_row())

    def test_negative_stop_pct_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S2Row.model_validate(_s2_row(stop_pct=-0.1))

    def test_negative_trades_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S2Row.model_validate(_s2_row(is_trades=-1))

    def test_missing_signal_method_rejected(self) -> None:
        bad = _s2_row()
        del bad["signal_method"]
        with pytest.raises(Exception):  # noqa: B017
            S2Row.model_validate(bad)


class TestS3Row:
    def test_valid(self) -> None:
        S3Row.model_validate(_s3_row())

    def test_n_test_periods_must_be_ge_1(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S3Row.model_validate(_s3_row(n_test_periods=0))

    def test_std_must_be_non_negative(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S3Row.model_validate(_s3_row(std_test_sharpe=-0.1))


class TestS4Row:
    def test_valid_minimal(self) -> None:
        S4Row.model_validate(_s4_row())

    def test_unknown_tier_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S4Row.model_validate(_s4_row(tier="NOT_A_TIER"))

    def test_psr_value_must_be_in_unit_interval(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S4Row.model_validate(_s4_row(psr_value=1.5))
        with pytest.raises(Exception):  # noqa: B017
            S4Row.model_validate(_s4_row(psr_value=-0.1))

    def test_psr_pass_flags_optional(self) -> None:
        """The CLI cohort selector flags may be absent."""
        S4Row.model_validate(_s4_row())

    def test_psr_pass_flag_when_present_is_bool(self) -> None:
        S4Row.model_validate(_s4_row(psr_pass=True, psr_pass_strict=False))


class TestS5Row:
    def test_valid(self) -> None:
        S5Row.model_validate(_s5_row())

    def test_oos_max_dd_must_be_non_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S5Row.model_validate(_s5_row(oos_max_dd=0.05))

    def test_unknown_cohort_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S5Row.model_validate(_s5_row(cohort="weekly"))

    def test_n_units_must_be_ge_1(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            S5Row.model_validate(_s5_row(n_units=0))


# ---------------------------------------------------------------------------
# validate_dataframe — strict vs accumulated errors
# ---------------------------------------------------------------------------


class TestValidateDataframe:
    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown stage"):
            validate_dataframe(pd.DataFrame(), "s99")

    def test_empty_df_passes(self) -> None:
        """Empty DataFrame returns unchanged — schema validation no-op."""
        df = pd.DataFrame()
        out = validate_dataframe(df, "s1")
        assert out is df

    def test_valid_rows_pass(self) -> None:
        df = pd.DataFrame([_s1_row(), _s1_row(pair="C/D", numerator="C",
                                              denominator="D")])
        out = validate_dataframe(df, "s1")
        assert out is df

    def test_strict_mode_raises_on_first_bad_row(self) -> None:
        df = pd.DataFrame([
            _s1_row(),
            _s1_row(window=0),   # invalid
            _s1_row(window=-5),  # also invalid — would be flagged in non-strict
        ])
        with pytest.raises(SchemaError, match="row 1"):
            validate_dataframe(df, "s1", strict=True)

    def test_non_strict_accumulates_up_to_max_errors(self) -> None:
        df = pd.DataFrame([
            _s1_row(window=0),
            _s1_row(window=-1),
            _s1_row(window=-2),
            _s1_row(window=-3),
            _s1_row(window=-4),
            _s1_row(window=-5),
        ])
        with pytest.raises(SchemaError) as exc_info:
            validate_dataframe(df, "s1", strict=False, max_errors=3)
        # Error message should reference at most 3 rows
        assert "3 row(s) failed" in str(exc_info.value)

    def test_non_strict_collects_all_when_under_cap(self) -> None:
        df = pd.DataFrame([_s1_row(window=0), _s1_row(window=-1)])
        with pytest.raises(SchemaError, match="2 row\\(s\\) failed"):
            validate_dataframe(df, "s1", max_errors=10)

    def test_returns_same_df_unchanged(self) -> None:
        df = pd.DataFrame([_s5_row()])
        out = validate_dataframe(df, "s5")
        assert out is df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_schema_for_known_stage(self) -> None:
        assert schema_for("s1") is S1Row
        assert schema_for("s5") is S5Row

    def test_schema_for_unknown_stage(self) -> None:
        with pytest.raises(SchemaError, match="Unknown stage"):
            schema_for("s99")

    def test_list_stages(self) -> None:
        assert list_stages() == ["s1", "s2", "s3", "s4", "s5"]

    def test_schema_version_is_set(self) -> None:
        assert isinstance(__schema_version__, str)
        assert __schema_version__.count(".") == 2  # semver
