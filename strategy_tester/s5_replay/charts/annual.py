"""Annual returns bar + quantstats-style snapshot + log equity.

Each chart exposes both a `*_figure()` (returns `go.Figure`) for the
JS-picker mega-report and a `*_html()` wrapper for direct embedding.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)
from strategy_tester.s5_replay.extra_metrics import annual_returns_series


# ---------------------------------------------------------------------------
# Annual bar
# ---------------------------------------------------------------------------


def annual_returns_bar_figure(
    eq: pd.Series, bench_eq: pd.Series | None, label: str,
) -> go.Figure:
    s = annual_returns_series(eq)
    fig = go.Figure()
    if s.empty:
        fig.update_layout(**dark_layout("Annual returns — (no data)", height=360))
        return fig
    b = (
        annual_returns_series(bench_eq) if bench_eq is not None
        else pd.Series(dtype=float)
    )
    fig.add_trace(go.Bar(
        x=s.index.astype(str), y=s.values * 100, name=label,
        marker_color=[
            "#3fb950" if v > 0 else "#ef5350" for v in s.values
        ],
    ))
    if not b.empty:
        b_aligned = b.reindex(s.index)
        fig.add_trace(go.Bar(
            x=b_aligned.index.astype(str), y=b_aligned.values * 100,
            name="SPY", marker_color="#9ecbff", opacity=0.55,
        ))
    mean = float(s.mean()) * 100
    fig.add_hline(
        y=mean, line=dict(color="#ffa726", dash="dash", width=1.4),
        annotation_text=f"mean={mean:+.1f}%",
        annotation_position="top right",
        annotation_font_color="#ffa726",
    )
    fig.add_hline(y=0, line=dict(color="#444", dash="solid", width=1))
    fig.update_layout(**dark_layout("Annual returns", height=360))
    fig.update_yaxes(title_text="Return (%)")
    fig.update_xaxes(title_text="Year", type="category")
    fig.update_layout(barmode="group")
    grid_axes_update(fig)
    return fig


def annual_returns_bar_html(
    eq: pd.Series, bench_eq: pd.Series | None, label: str, div_id: str,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = annual_returns_bar_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Log equity
# ---------------------------------------------------------------------------


def log_equity_figure(
    eq: pd.Series, bench_eq: pd.Series | None, label: str,
) -> go.Figure:
    fig = go.Figure()
    if eq.empty:
        fig.update_layout(**dark_layout("Cumulative return — log scale", height=360))
        return fig
    cum = eq / eq.iloc[0]
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, mode="lines", name=label,
        line=dict(color="#58a6ff", width=2),
    ))
    if bench_eq is not None and not bench_eq.empty:
        bc = bench_eq / bench_eq.iloc[0]
        fig.add_trace(go.Scatter(
            x=bc.index, y=bc.values, mode="lines", name="SPY",
            line=dict(color="#ef5350", dash="dash", width=1.4),
        ))
    fig.update_layout(**dark_layout("Cumulative return — log scale", height=360))
    fig.update_yaxes(type="log", title_text="Cum. return (1.0 = start)")
    grid_axes_update(fig)
    return fig


def log_equity_html(
    eq: pd.Series, bench_eq: pd.Series | None, label: str, div_id: str,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = log_equity_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Snapshot 2-panel (cum + drawdown)
# ---------------------------------------------------------------------------


def snapshot_3panel_figure(eq: pd.Series, label: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.05,
        subplot_titles=("Cumulative return (%)", "Drawdown (%)"),
    )
    if eq.empty:
        fig.update_layout(**dark_layout(f"Snapshot — {label}", height=520))
        return fig
    cum = eq / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1) * 100
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values * 100, mode="lines", name=label,
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="DD",
        line=dict(color="#ef5350", width=1.4),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.18)",
        showlegend=False,
    ), row=2, col=1)
    fig.update_layout(**dark_layout(f"Snapshot — {label}", height=520))
    fig.update_yaxes(title_text="Cum (%)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    grid_axes_update(fig)
    return fig


def snapshot_3panel_html(eq: pd.Series, label: str, div_id: str) -> str:
    """Backwards-compat: function name retained even though it's 2-panel now."""
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = snapshot_3panel_figure(eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
