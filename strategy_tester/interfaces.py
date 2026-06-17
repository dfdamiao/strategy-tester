"""Stage interface contracts — required columns per boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# --- Required columns per stage output ---

S1_REQUIRED = frozenset({
    "pair", "numerator", "denominator", "passed",
    "halflife", "window", "method",
})

S2_REQUIRED = frozenset({
    "pair", "numerator", "denominator", "halflife", "window",
    "entry_thresh", "exit_thresh", "stop_pct", "slope_min",
    "is_sharpe", "is_penalized_sharpe", "is_trades",
    "oos_sharpe", "oos_trades", "passed",
    "signal_method", "optim_method",
})

S3_REQUIRED = frozenset({
    "pair", "numerator", "denominator",
    "mean_test_sharpe", "std_test_sharpe", "n_test_periods",
    "baseline_sharpe", "degradation", "passed", "val_method",
})

S4_REQUIRED = frozenset({
    "pair", "numerator", "denominator", "passed", "tier",
})

_STAGE_COLUMNS = {
    "s1": S1_REQUIRED,
    "s2": S2_REQUIRED,
    "s3": S3_REQUIRED,
    "s4": S4_REQUIRED,
}


def validate_interface(
    df: pd.DataFrame,
    stage: str,
) -> None:
    """Raise ValueError if required columns missing."""
    if stage not in _STAGE_COLUMNS:
        raise ValueError(f"Unknown stage {stage!r}")
    required = _STAGE_COLUMNS[stage]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for {stage}: "
            f"{sorted(missing)}"
        )


@dataclass
class PipelineResult:
    """Result container returned by Pipeline.run()."""

    name: str
    stages: dict[str, dict[str, pd.DataFrame]] = field(
        default_factory=dict
    )
    reports: dict[str, Path] = field(default_factory=dict)
    final: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
