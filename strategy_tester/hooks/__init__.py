"""Pre/post-stage hook system for cross-cutting concerns.

Hook events (matches MODULARITY.md §5):
  pre_pipeline / post_pipeline       — wrap a full run
  pre_stage / post_stage             — wrap a single stage
  on_anomaly                         — fired when an AnomalyResult is emitted
  on_checkpoint_save / on_checkpoint_load — checkpoint lifecycle

Hooks receive a ``PipelineContext`` (passed by lib.pipeline at run time).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from strategy_tester.anomaly import AnomalyResult

HookEvent = Literal[
    "pre_pipeline",
    "post_pipeline",
    "pre_stage",
    "post_stage",
    "on_anomaly",
    "on_checkpoint_save",
    "on_checkpoint_load",
]

_HOOKS: dict[str, list[tuple[dict[str, Any], Callable[..., Any]]]] = {
    "pre_pipeline": [],
    "post_pipeline": [],
    "pre_stage": [],
    "post_stage": [],
    "on_anomaly": [],
    "on_checkpoint_save": [],
    "on_checkpoint_load": [],
}


@dataclass
class PipelineContext:
    """Mutable context passed to hooks."""

    strategy: str
    config: dict[str, Any]
    output_dir: Path
    logger: logging.Logger
    run_log: list[str] = field(default_factory=list)
    run_id: str = ""
    anomalies: list[AnomalyResult] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def register_hook(
    event: HookEvent,
    *,
    stage: str | None = None,
    severity: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a hook for an event.

    ``stage`` filter applies to pre_stage / post_stage hooks.
    ``severity`` filter applies to on_anomaly hooks.
    """
    if event not in _HOOKS:
        valid = sorted(_HOOKS)
        raise ValueError(f"Unknown event {event!r}. Valid: {valid}")

    filters: dict[str, Any] = {}
    if stage is not None:
        filters["stage"] = stage
    if severity is not None:
        filters["severity"] = severity

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _HOOKS[event].append((filters, fn))
        return fn

    return decorator


def fire(event: HookEvent, **kwargs: Any) -> None:
    """Fire all registered hooks for an event, filtered by kwargs."""
    if event not in _HOOKS:
        raise ValueError(f"Unknown event {event!r}")
    for filters, fn in _HOOKS[event]:
        if not _matches_filters(filters, kwargs):
            continue
        try:
            fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Hook errors must not break the pipeline.
            ctx = kwargs.get("ctx")
            if ctx is not None:
                ctx.logger.warning(
                    f"Hook {fn.__name__!r} failed for {event}: {exc}"
                )


def _matches_filters(filters: dict[str, Any], kwargs: dict[str, Any]) -> bool:
    for key, want in filters.items():
        got = kwargs.get(key)
        # AnomalyResult.severity for on_anomaly
        if (
            key == "severity"
            and isinstance(kwargs.get("anomaly"), AnomalyResult)
        ):
            got = kwargs["anomaly"].severity
        if got != want:
            return False
    return True


def clear_hooks() -> None:
    """Reset all registered hooks. Used by tests."""
    for event in _HOOKS:
        _HOOKS[event].clear()


def list_hooks() -> dict[str, list[str]]:
    """List registered hooks per event (for --help)."""
    return {
        event: [fn.__name__ for _filters, fn in hooks]
        for event, hooks in _HOOKS.items()
    }


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


@register_hook("post_stage")
def _log_stage_summary(
    ctx: PipelineContext,
    stage: str,
    output_df: pd.DataFrame,
) -> None:
    """Append a one-line summary to ctx.run_log."""
    n_rows = len(output_df)
    cols = list(output_df.columns)
    if "passed" in cols:
        n_passed = int(output_df["passed"].sum())
        ctx.run_log.append(
            f"[{stage}] {n_passed}/{n_rows} passed gate"
        )
    else:
        ctx.run_log.append(f"[{stage}] {n_rows} rows")


@register_hook("on_anomaly")
def _record_anomaly(ctx: PipelineContext, anomaly: AnomalyResult) -> None:
    """Record every anomaly in ctx.anomalies + jsonl trail."""
    ctx.anomalies.append(anomaly)
    ctx.logger.warning(
        f"[anomaly:{anomaly.severity}] {anomaly.stage}/{anomaly.check}: "
        f"{anomaly.message}"
    )
    # Append to jsonl trail
    jsonl_path = ctx.output_dir / "anomalies.jsonl"
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with jsonl_path.open("a") as fh:
            fh.write(json.dumps(anomaly.to_dict()) + "\n")
    except OSError as exc:
        ctx.logger.warning(f"could not write anomaly to {jsonl_path}: {exc}")
