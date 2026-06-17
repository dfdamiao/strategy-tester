"""HTML report generator (stub — extend with plotly as needed)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_tester.interfaces import PipelineResult


def generate_stage_report(
    result: PipelineResult,
    output_dir: Path,
) -> Path:
    """Generate per-stage HTML summary. Returns path to report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{result.name}_report.html"

    sections = []
    for stage, data in result.stages.items():
        df = data.get("result")
        if df is not None and isinstance(df, pd.DataFrame):
            n_rows = len(df)
            n_pass = int(df["passed"].sum()) if "passed" in df.columns else 0
            sections.append(
                f"<h2>{stage.upper()}</h2>"
                f"<p>Rows: {n_rows} | Passed: {n_pass}</p>"
                f"{df.head(20).to_html()}"
            )

    html = (
        f"<html><head><title>{result.name}</title></head>"
        f"<body><h1>{result.name}</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    html_path.write_text(html)
    return html_path
