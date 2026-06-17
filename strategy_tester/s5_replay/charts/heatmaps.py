"""Monthly returns heatmap + cross-scheme correlation heatmap.

Exposes both *_figure() and *_html() builders.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)
from strategy_tester.s5_replay.extra_metrics import monthly_returns_table


# ---------------------------------------------------------------------------
# Monthly heatmap
# ---------------------------------------------------------------------------


def monthly_heatmap_figure(eq: pd.Series, title: str) -> go.Figure:
    pivot = monthly_returns_table(eq)
    fig = go.Figure()
    if pivot.empty:
        fig.update_layout(**dark_layout(title + " (no monthly data)", height=320))
        return fig
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = [c for c in months if c in pivot.columns]
    z = (pivot[cols] * 100).values
    text = [[f"{v:+.1f}" if pd.notna(v) else "" for v in row] for row in z]
    fig.add_trace(go.Heatmap(
        z=z,
        x=cols,
        y=[str(y) for y in pivot.index],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#e6edf3"},
        colorscale=[
            [0.0, "#5b1e1e"],
            [0.4, "#a13030"],
            [0.49, "#3f1d1d"],
            [0.5, "#16191c"],
            [0.51, "#1c3a1c"],
            [0.6, "#2ea043"],
            [1.0, "#3fb950"],
        ],
        zmid=0,
        colorbar=dict(title="%"),
    ))
    fig.update_layout(**dark_layout(
        title, height=max(280, 28 * len(pivot.index) + 100),
    ))
    fig.update_xaxes(side="top")
    grid_axes_update(fig)
    return fig


def monthly_heatmap_html(eq: pd.Series, title: str, div_id: str) -> str:
    fig = monthly_heatmap_figure(eq, title)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Cross-scheme correlation heatmap (SPY + buy-hold + schemes)
# ---------------------------------------------------------------------------


def scheme_correlation_heatmap_html(
    equity_by_scheme: dict[str, pd.Series],
    title: str,
    div_id: str,
    max_schemes: int = 30,
    benchmark_eq: pd.Series | None = None,
    benchmark_name: str = "SPY",
    bh_eq: pd.Series | None = None,
    bh_name: str = "Buy-Hold",
) -> str:
    """(SPY + BH + N) × (SPY + BH + N) correlation heatmap."""
    schemes = list(equity_by_scheme.keys())[:max_schemes]
    if len(schemes) < 1:
        return f"<div id='{div_id}'>(no schemes)</div>"

    series_list: list[pd.Series] = []
    labels: list[str] = []
    label_is_bench: list[bool] = []
    if benchmark_eq is not None and len(benchmark_eq) > 1:
        series_list.append(benchmark_eq.pct_change().rename(benchmark_name))
        labels.append(benchmark_name)
        label_is_bench.append(True)
    if bh_eq is not None and len(bh_eq) > 1:
        series_list.append(bh_eq.pct_change().rename(bh_name))
        labels.append(bh_name)
        label_is_bench.append(True)
    for s in schemes:
        series_list.append(equity_by_scheme[s].pct_change().rename(s))
        labels.append(s)
        label_is_bench.append(False)

    if len(labels) < 2:
        return f"<div id='{div_id}'>(need 2+ series)</div>"

    rets = pd.concat(series_list, axis=1, join="inner").dropna()
    if rets.empty:
        return f"<div id='{div_id}'>(no overlapping data)</div>"
    corr = rets.corr().values
    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=[
            [0.0, "#a13030"],
            [0.5, "#16191c"],
            [1.0, "#3fb950"],
        ],
        colorbar=dict(title="ρ"),
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        textfont={"size": 9, "color": "#e6edf3"},
    ))
    fig.update_layout(**dark_layout(title, height=max(360, 22 * len(labels) + 120)))
    fig.update_xaxes(tickangle=-45, side="bottom")
    tick_text = [
        f"<span style='color:#ef5350;font-weight:700'>{lab}</span>"
        if is_b else lab
        for lab, is_b in zip(labels, label_is_bench, strict=False)
    ]
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(labels))),
                     ticktext=tick_text)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))),
                     ticktext=tick_text)
    grid_axes_update(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


__all__ = [
    "monthly_heatmap_figure",
    "monthly_heatmap_html",
    "scheme_correlation_heatmap_html",
]
