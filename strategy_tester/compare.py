"""Compare multiple pipelines on same data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_tester.interfaces import PipelineResult
from strategy_tester.pipeline import Pipeline


def compare_pipelines(
    pipelines: dict[str, "Pipeline | dict"],
    prices: pd.DataFrame,
    pairs: list[dict],
    config: dict | None = None,
    output_dir: Path | None = None,
) -> dict[str, PipelineResult]:
    """Run all pipelines on same data. Returns {name: PipelineResult}."""
    results: dict[str, PipelineResult] = {}
    for name, pipe in pipelines.items():
        if isinstance(pipe, dict):
            pipe = Pipeline(**pipe)
        results[name] = pipe.run(prices, pairs, config)
    return results
