"""End-to-end integration test for lib.{anomaly, hooks, io, schemas}.

Wires the four new modules together on synthetic S1-S5 outputs to verify
the parity contract referenced in schemas.py:
  Phase 1.E foundation (2026-05-17). Not yet wired into Pipeline.run —
  migration is gated on the post-kickoff parity audit (Phase 1.C).

This test simulates one full pipeline tick:
  1. Produce a stage DataFrame
  2. Validate against schemas + write via lib.io
  3. Read back, run anomaly checks on the round-tripped frame
  4. Fire on_anomaly via hooks, collect them into PipelineContext
  5. Assert end-state: jsonl trail + run_log + ctx.anomalies all coherent

Marked `integration` (slower than unit; still <1s).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from strategy_tester.anomaly import AnomalyResult, run_checks
from strategy_tester.hooks import (
    PipelineContext,
    _HOOKS,
    fire,
)
from strategy_tester.io import (
    build_tag,
    read_stage_output,
    write_stage_output,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_hooks():
    saved = {event: list(hooks) for event, hooks in _HOOKS.items()}
    yield
    for event in _HOOKS:
        _HOOKS[event].clear()
        _HOOKS[event].extend(saved[event])


@pytest.fixture
def ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        strategy="framework_int_test",
        config={"s4_psr": 0.95},
        output_dir=tmp_path,
        logger=logging.getLogger("framework_int_test"),
    )


@pytest.fixture
def s1_clean() -> pd.DataFrame:
    """50 rows, 40% pass — inside (0.05, 0.95) band → no anomaly."""
    return pd.DataFrame([
        {
            "pair": f"A{i}/B", "numerator": f"A{i}", "denominator": "B",
            "passed": i < 20,
            "halflife": 30.0, "window": 15,
            "method": "chan_halflife",
        }
        for i in range(50)
    ])


@pytest.fixture
def s4_bad() -> pd.DataFrame:
    """1% pass — under floor (0.05) → triggers s4_pass_rate STRICT anomaly."""
    return pd.DataFrame([
        {
            "pair": f"A{i}/B", "numerator": f"A{i}", "denominator": "B",
            "passed": i == 0,
            "tier": "GOOD" if i == 0 else "REJECT",
        }
        for i in range(100)
    ])


# ---------------------------------------------------------------------------
# Tier 1 — happy path (clean S1 round-trip + no anomalies)
# ---------------------------------------------------------------------------


def test_s1_clean_round_trip_no_anomalies(
    tmp_path: Path,
    ctx: PipelineContext,
    s1_clean: pd.DataFrame,
    snapshot_hooks,
) -> None:
    """Schema-valid S1 → write+read parity → no anomaly checks fire."""
    tag = build_tag(ctx.config)
    path = write_stage_output(s1_clean, "s1", tmp_path, tag=tag)
    assert path.name == "s1_psr95_metrics.parquet"

    back = read_stage_output(path, "s1")
    pd.testing.assert_frame_equal(back, s1_clean, check_like=True)

    anomalies = run_checks("s1", back)
    assert anomalies == []

    for a in anomalies:
        fire("on_anomaly", ctx=ctx, anomaly=a)

    assert ctx.anomalies == []
    assert not (ctx.output_dir / "anomalies.jsonl").exists()


# ---------------------------------------------------------------------------
# Tier 2 — anomaly path (S4 pass rate too low → strict anomaly fires)
# ---------------------------------------------------------------------------


def test_s4_bad_pass_rate_triggers_strict_anomaly(
    tmp_path: Path,
    ctx: PipelineContext,
    s4_bad: pd.DataFrame,
    snapshot_hooks,
) -> None:
    """S4 with 1% pass rate must trigger a strict anomaly and persist to jsonl."""
    path = write_stage_output(s4_bad, "s4", tmp_path)
    back = read_stage_output(path, "s4")

    anomalies = run_checks("s4", back)
    s4_anoms = [a for a in anomalies if a.check == "s4_pass_rate"]
    assert len(s4_anoms) == 1, (
        f"Expected 1 s4_pass_rate anomaly, got {len(s4_anoms)} "
        f"(all: {[a.check for a in anomalies]})"
    )
    assert s4_anoms[0].severity == "strict"

    for a in anomalies:
        fire("on_anomaly", ctx=ctx, anomaly=a)

    # ctx.anomalies populated by _record_anomaly built-in
    assert s4_anoms[0] in ctx.anomalies

    # jsonl trail persisted
    jsonl = ctx.output_dir / "anomalies.jsonl"
    assert jsonl.exists()
    records = [
        json.loads(line) for line in jsonl.read_text().strip().splitlines()
    ]
    assert any(r["check"] == "s4_pass_rate" for r in records)
    assert any(r["severity"] == "strict" for r in records)


# ---------------------------------------------------------------------------
# Tier 3 — post_stage built-in records run_log line on every stage write
# ---------------------------------------------------------------------------


def test_post_stage_built_in_logs_summary(
    ctx: PipelineContext,
    s1_clean: pd.DataFrame,
    s4_bad: pd.DataFrame,
    snapshot_hooks,
) -> None:
    """_log_stage_summary (built-in) appends one line per post_stage fire."""
    fire("post_stage", ctx=ctx, stage="s1", output_df=s1_clean)
    fire("post_stage", ctx=ctx, stage="s4", output_df=s4_bad)
    assert any("[s1]" in line and "20/50" in line for line in ctx.run_log)
    assert any("[s4]" in line and "1/100" in line for line in ctx.run_log)


# ---------------------------------------------------------------------------
# Tier 4 — schema validation rejects bad rows BEFORE they reach anomaly path
# ---------------------------------------------------------------------------


def test_schema_rejects_invalid_tier_before_io_write(
    tmp_path: Path,
) -> None:
    """write_stage_output validates via lib.schemas — invalid Literal rejected."""
    bad = pd.DataFrame([{
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "passed": True, "tier": "EXCELLENT",  # not in S4Row Literal
    }])
    from strategy_tester.schemas import SchemaError

    with pytest.raises(SchemaError):
        write_stage_output(bad, "s4", tmp_path)


# ---------------------------------------------------------------------------
# Tier 5 — full S1→S5 cascade: every stage hits write → read → check
# ---------------------------------------------------------------------------


def test_full_cascade_s1_to_s5(
    tmp_path: Path,
    ctx: PipelineContext,
    snapshot_hooks,
) -> None:
    """One bar through every stage. Verifies stage-to-stage independence."""
    stages: dict[str, pd.DataFrame] = {
        "s1": pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "halflife": 30.0, "window": 15,
            "method": "chan_halflife",
        }]),
        "s2": pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "halflife": 30.0, "window": 15,
            "entry_thresh": -2.0, "exit_thresh": 0.5, "stop_pct": 0.1,
            "slope_min": 0.0, "is_sharpe": 1.2, "is_penalized_sharpe": 1.0,
            "is_trades": 50, "oos_sharpe": 0.8, "oos_trades": 10,
            "passed": True, "signal_method": "zscore_robust_mad",
            "optim_method": "grid_search",
        }]),
        "s3": pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "mean_test_sharpe": 0.7, "std_test_sharpe": 0.3,
            "n_test_periods": 8, "baseline_sharpe": 1.0,
            "degradation": 0.3, "passed": True, "val_method": "wfa_expanding",
        }]),
        "s4": pd.DataFrame([{
            "pair": "A/B", "numerator": "A", "denominator": "B",
            "passed": True, "tier": "GOOD",
        }]),
        "s5": pd.DataFrame([{
            "scheme_name": "equal_weight_flat_1k_cap10",
            "cohort": "ratios",
            "oos_sr_full": 1.2, "oos_cagr": 0.18, "oos_max_dd": -0.15,
            "n_units": 30,
        }]),
    }

    captured: list[tuple[str, AnomalyResult]] = []
    for stage, df in stages.items():
        path = write_stage_output(df, stage, tmp_path)
        back = read_stage_output(path, stage)
        assert len(back) == 1, f"{stage}: round-trip lost a row"

        # No anomalies expected on this single-row clean cohort (cohort_size
        # check will fire on s3 because passed_count=1 < 50 — that's expected).
        anomalies = run_checks(stage, back)
        for a in anomalies:
            captured.append((stage, a))
            fire("on_anomaly", ctx=ctx, anomaly=a)
        fire("post_stage", ctx=ctx, stage=stage, output_df=back)

    # s3_cohort_size MUST fire (1 < 50)
    fired_checks = {a.check for _, a in captured}
    assert "s3_cohort_size" in fired_checks

    # Every stage produced a run_log line
    assert sum(1 for line in ctx.run_log if line.startswith("[s")) == 5
