"""Tests for lib.hooks — register / fire / filter / hook-error isolation.

Built-in hooks (registered at import time):
  _log_stage_summary  → post_stage
  _record_anomaly     → on_anomaly
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from strategy_tester.anomaly import AnomalyResult
from strategy_tester.hooks import (
    PipelineContext,
    _HOOKS,
    clear_hooks,
    fire,
    list_hooks,
    register_hook,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        strategy="test_strategy",
        config={},
        output_dir=tmp_path,
        logger=logging.getLogger("test_hooks"),
    )


@pytest.fixture
def snapshot_hooks():
    """Snapshot the global _HOOKS state and restore after each test.

    The built-in hooks (_log_stage_summary, _record_anomaly) register at
    import time — naive clear_hooks() would wipe them. This fixture preserves
    them across tests that need to add and clear custom hooks.
    """
    saved = {event: list(hooks) for event, hooks in _HOOKS.items()}
    yield
    for event in _HOOKS:
        _HOOKS[event].clear()
        _HOOKS[event].extend(saved[event])


# ---------------------------------------------------------------------------
# register_hook / fire
# ---------------------------------------------------------------------------


class TestRegisterAndFire:
    def test_unknown_event_raises(self, snapshot_hooks) -> None:
        with pytest.raises(ValueError, match="Unknown event"):
            register_hook("not_an_event")(lambda **_: None)  # type: ignore[arg-type]

    def test_registered_hook_fires(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        calls: list[dict[str, object]] = []

        @register_hook("pre_pipeline")
        def _h(ctx: PipelineContext, **_: object) -> None:
            calls.append({"strategy": ctx.strategy})

        fire("pre_pipeline", ctx=ctx)
        assert calls == [{"strategy": "test_strategy"}]

    def test_fire_unknown_event_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown event"):
            fire("bogus_event")  # type: ignore[arg-type]

    def test_multiple_hooks_fire_in_registration_order(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        order: list[str] = []

        @register_hook("pre_pipeline")
        def _first(**_: object) -> None:
            order.append("first")

        @register_hook("pre_pipeline")
        def _second(**_: object) -> None:
            order.append("second")

        fire("pre_pipeline", ctx=ctx)
        assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# Filter matching — stage / severity
# ---------------------------------------------------------------------------


class TestFilterMatching:
    def test_stage_filter_matches_only_target_stage(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        calls: list[str] = []

        @register_hook("post_stage", stage="s4")
        def _only_s4(stage: str, **_: object) -> None:
            calls.append(stage)

        df = pd.DataFrame({"passed": [True, False]})
        fire("post_stage", ctx=ctx, stage="s4", output_df=df)
        fire("post_stage", ctx=ctx, stage="s1", output_df=df)
        assert calls == ["s4"]

    def test_severity_filter_on_anomaly(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        """severity= filter reads AnomalyResult.severity, not kwargs['severity']."""
        captured: list[AnomalyResult] = []

        @register_hook("on_anomaly", severity="strict")
        def _only_strict(anomaly: AnomalyResult, **_: object) -> None:
            captured.append(anomaly)

        warn_anom = AnomalyResult(
            stage="s1", check="s1_pass_rate", value=0.02,
            threshold=0.05, severity="warn", message="warn",
        )
        strict_anom = AnomalyResult(
            stage="s2", check="s2_oos_sr", value=3.0,
            threshold=2.0, severity="strict", message="strict",
        )
        fire("on_anomaly", ctx=ctx, anomaly=warn_anom)
        fire("on_anomaly", ctx=ctx, anomaly=strict_anom)
        assert captured == [strict_anom]


# ---------------------------------------------------------------------------
# Hook-error isolation — broken hook must not break pipeline
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_hook_exception_does_not_propagate(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        @register_hook("pre_pipeline")
        def _broken(**_: object) -> None:
            raise RuntimeError("boom")

        # Should not raise — exception swallowed and logged
        fire("pre_pipeline", ctx=ctx)

    def test_broken_hook_does_not_block_subsequent_hooks(
        self, ctx: PipelineContext, snapshot_hooks,
    ) -> None:
        survived: list[bool] = []

        @register_hook("pre_pipeline")
        def _broken(**_: object) -> None:
            raise RuntimeError("boom")

        @register_hook("pre_pipeline")
        def _later(**_: object) -> None:
            survived.append(True)

        fire("pre_pipeline", ctx=ctx)
        assert survived == [True]


# ---------------------------------------------------------------------------
# clear_hooks / list_hooks
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_clear_hooks_removes_all(self, snapshot_hooks) -> None:
        @register_hook("pre_pipeline")
        def _h(**_: object) -> None: ...

        clear_hooks()
        for event_hooks in _HOOKS.values():
            assert event_hooks == []

    def test_list_hooks_returns_names_per_event(
        self, snapshot_hooks,
    ) -> None:
        listing = list_hooks()
        # Built-in hooks should be present
        assert "_log_stage_summary" in listing["post_stage"]
        assert "_record_anomaly" in listing["on_anomaly"]


# ---------------------------------------------------------------------------
# Built-in: _log_stage_summary
# ---------------------------------------------------------------------------


class TestLogStageSummary:
    def test_summary_with_passed_column(
        self, ctx: PipelineContext,
    ) -> None:
        df = pd.DataFrame({"passed": [True, True, False, True]})
        fire("post_stage", ctx=ctx, stage="s4", output_df=df)
        # The built-in records "[s4] 3/4 passed gate" style line
        assert any("[s4]" in line and "3/4" in line for line in ctx.run_log)

    def test_summary_without_passed_column(
        self, ctx: PipelineContext,
    ) -> None:
        df = pd.DataFrame({"oos_sr_full": [1.2, 0.8]})
        fire("post_stage", ctx=ctx, stage="s5", output_df=df)
        assert any("[s5] 2 rows" in line for line in ctx.run_log)


# ---------------------------------------------------------------------------
# Built-in: _record_anomaly
# ---------------------------------------------------------------------------


class TestRecordAnomaly:
    def test_anomaly_recorded_on_context(
        self, ctx: PipelineContext,
    ) -> None:
        anom = AnomalyResult(
            stage="s1", check="s1_pass_rate", value=0.02,
            threshold=0.05, severity="warn", message="pass rate low",
        )
        fire("on_anomaly", ctx=ctx, anomaly=anom)
        assert anom in ctx.anomalies

    def test_anomaly_appended_to_jsonl_trail(
        self, ctx: PipelineContext,
    ) -> None:
        """jsonl trail one row per anomaly, ndjson format."""
        anom1 = AnomalyResult(
            stage="s1", check="s1_pass_rate", value=0.02,
            threshold=0.05, severity="warn", message="m1",
        )
        anom2 = AnomalyResult(
            stage="s4", check="s4_pass_rate", value=0.01,
            threshold=0.05, severity="strict", message="m2",
        )
        fire("on_anomaly", ctx=ctx, anomaly=anom1)
        fire("on_anomaly", ctx=ctx, anomaly=anom2)

        jsonl = ctx.output_dir / "anomalies.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        rec2 = json.loads(lines[1])
        assert rec1["check"] == "s1_pass_rate"
        assert rec2["check"] == "s4_pass_rate"
        assert rec2["severity"] == "strict"

    def test_multiple_fires_append_not_overwrite(
        self, ctx: PipelineContext,
    ) -> None:
        for i in range(5):
            anom = AnomalyResult(
                stage="s1", check="s1_pass_rate", value=float(i),
                threshold=0.05, severity="warn", message=f"#{i}",
            )
            fire("on_anomaly", ctx=ctx, anomaly=anom)

        jsonl = ctx.output_dir / "anomalies.jsonl"
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 5
