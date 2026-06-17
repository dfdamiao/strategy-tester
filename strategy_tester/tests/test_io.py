"""Tests for lib.io — schema-validated read/write + tag-building convention.

File naming contract (CLI_REFERENCE.md §11):
  {stage}[_{cohort}][_{tag}]_metrics.{ext}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from strategy_tester.io import (
    build_tag,
    read_stage_output,
    write_stage_output,
)
from strategy_tester.schemas import SchemaError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures — minimal valid rows per stage
# ---------------------------------------------------------------------------


@pytest.fixture
def s4_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "tier": "GOOD",
        },
        {
            "pair": "C/D", "numerator": "C", "denominator": "D",
            "passed": False, "tier": "REJECT",
        },
    ])


@pytest.fixture
def s5_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scheme_name": "equal_weight_flat_1k_cap10",
            "cohort": "ratios",
            "oos_sr_full": 1.2, "oos_cagr": 0.18, "oos_max_dd": -0.15,
            "n_units": 30,
        },
    ])


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------


class TestFilenameConvention:
    def test_stage_only(self, tmp_path: Path, s4_df: pd.DataFrame) -> None:
        path = write_stage_output(s4_df, "s4", tmp_path)
        assert path.name == "s4_metrics.parquet"

    def test_stage_plus_cohort(self, tmp_path: Path, s5_df: pd.DataFrame) -> None:
        path = write_stage_output(s5_df, "s5", tmp_path, cohort="ratios")
        assert path.name == "s5_ratios_metrics.parquet"

    def test_stage_plus_cohort_plus_tag(
        self, tmp_path: Path, s5_df: pd.DataFrame,
    ) -> None:
        path = write_stage_output(
            s5_df, "s5", tmp_path, cohort="ratios", tag="psr95",
        )
        assert path.name == "s5_ratios_psr95_metrics.parquet"

    def test_csv_extension(self, tmp_path: Path, s4_df: pd.DataFrame) -> None:
        path = write_stage_output(s4_df, "s4", tmp_path, fmt="csv")
        assert path.suffix == ".csv"
        assert path.name == "s4_metrics.csv"

    def test_unsupported_format_raises(
        self, tmp_path: Path, s4_df: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            write_stage_output(s4_df, "s4", tmp_path, fmt="hdf5")


# ---------------------------------------------------------------------------
# Round-trip with validation
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_parquet_round_trip_preserves_data(
        self, tmp_path: Path, s4_df: pd.DataFrame,
    ) -> None:
        path = write_stage_output(s4_df, "s4", tmp_path)
        back = read_stage_output(path, "s4")
        pd.testing.assert_frame_equal(back, s4_df, check_like=True)

    def test_csv_round_trip(
        self, tmp_path: Path, s4_df: pd.DataFrame,
    ) -> None:
        path = write_stage_output(s4_df, "s4", tmp_path, fmt="csv")
        back = read_stage_output(path, "s4")
        pd.testing.assert_frame_equal(back, s4_df, check_like=True)

    def test_creates_missing_directory(
        self, tmp_path: Path, s4_df: pd.DataFrame,
    ) -> None:
        nested = tmp_path / "deeply" / "nested" / "out"
        path = write_stage_output(s4_df, "s4", nested)
        assert path.exists()
        assert nested.is_dir()

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_stage_output(tmp_path / "nope.parquet", "s4")

    def test_read_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "out.xlsx"
        bogus.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported extension"):
            read_stage_output(bogus, "s4")


# ---------------------------------------------------------------------------
# Schema validation on write/read
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_write_rejects_invalid_row(self, tmp_path: Path) -> None:
        bad = pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "tier": "INVALID_TIER",
        }])
        with pytest.raises(SchemaError):
            write_stage_output(bad, "s4", tmp_path)

    def test_validate_false_bypasses_schema_on_write(
        self, tmp_path: Path,
    ) -> None:
        """Escape hatch — preserves backward compat with old artifacts."""
        bad = pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "tier": "INVALID_TIER",
        }])
        path = write_stage_output(bad, "s4", tmp_path, validate=False)
        assert path.exists()

    def test_validate_false_bypasses_schema_on_read(
        self, tmp_path: Path,
    ) -> None:
        bad = pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "tier": "INVALID_TIER",
        }])
        path = write_stage_output(bad, "s4", tmp_path, validate=False)
        # Read with validate=True must reject; with validate=False must pass
        with pytest.raises(SchemaError):
            read_stage_output(path, "s4")
        df = read_stage_output(path, "s4", validate=False)
        assert df.iloc[0]["tier"] == "INVALID_TIER"


# ---------------------------------------------------------------------------
# build_tag — override → human-readable suffix
# ---------------------------------------------------------------------------


class TestBuildTag:
    def test_empty_overrides_empty_tag(self) -> None:
        assert build_tag({}) == ""

    def test_known_psr_key_shortened(self) -> None:
        assert build_tag({"s4_psr": 0.95}) == "psr95"

    def test_known_dsr_key_shortened(self) -> None:
        assert build_tag({"s4_dsr": 0.90}) == "dsr90"

    def test_keys_sorted_for_determinism(self) -> None:
        """Same overrides → same tag regardless of dict insertion order."""
        a = build_tag({"s4_psr": 0.95, "s4_dsr": 0.90})
        b = build_tag({"s4_dsr": 0.90, "s4_psr": 0.95})
        assert a == b

    def test_bool_serialized_as_on_off(self) -> None:
        on = build_tag({"s2_is_ratio": True})
        off = build_tag({"s2_is_ratio": False})
        assert on != off
        assert "on" in on
        assert "off" in off

    def test_int_passes_through(self) -> None:
        assert "s3folds8" in build_tag({"s3_folds": 8})

    def test_long_tag_falls_back_to_hash(self) -> None:
        """Tag longer than max_length collapses to short hex digest."""
        overrides: dict[str, object] = {f"key_{i}": float(i) for i in range(20)}
        tag = build_tag(overrides, max_length=20)
        # Falls back to an 8-char hex digest
        assert len(tag) == 8
        assert all(c in "0123456789abcdef" for c in tag)

    def test_hash_is_deterministic(self) -> None:
        overrides: dict[str, object] = {f"key_{i}": float(i) for i in range(20)}
        a = build_tag(overrides, max_length=20)
        b = build_tag(overrides, max_length=20)
        assert a == b

    def test_unknown_key_passthrough_strips_underscores(self) -> None:
        tag = build_tag({"unknown_key_name": 1})
        # Underscores in keys stripped per impl ("custom" key path)
        assert "_" not in tag.split("_")[0] if "_" in tag else True
        assert "unknownkeyname" in tag
