"""Bootstrap visualisations: forecast cone, perf-stats box plots, monte carlo.

Exposes both *_figure() for the picker and *_html() for direct embed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy_tester.s5_replay.bootstrap import (
    forecast_cone_bootstrap,
    perf_stats_bootstrap,
)
from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)


# ---------------------------------------------------------------------------
# Forecast cone
# ---------------------------------------------------------------------------


def forecast_cone_figure(
    eq: pd.Series, label: str, forward_days: int = 252,
    n_samples: int = 5000,
) -> go.Figure:
    fig = go.Figure()
    if len(eq) < 60:
        fig.update_layout(**dark_layout("Forecast cone (insufficient data)", height=460))
        return fig
    cone = forecast_cone_bootstrap(
        eq, forward_days=forward_days, n_samples=n_samples,
    )
    if cone.empty:
        fig.update_layout(**dark_layout("Forecast cone (calc failed)", height=460))
        return fig

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name=label,
        line=dict(color="#58a6ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cone.index, y=cone["upper_2sd"], mode="lines",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=cone.index, y=cone["lower_2sd"], mode="lines",
        fill="tonexty", fillcolor="rgba(63,185,80,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="2σ band",
    ))
    fig.add_trace(go.Scatter(
        x=cone.index, y=cone["upper_1sd"], mode="lines",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=cone.index, y=cone["lower_1sd"], mode="lines",
        fill="tonexty", fillcolor="rgba(63,185,80,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="1σ band",
    ))
    fig.add_trace(go.Scatter(
        x=cone.index, y=cone["median"], mode="lines", name="Median forecast",
        line=dict(color="#3fb950", width=2, dash="dot"),
    ))
    fig.update_layout(**dark_layout(
        f"Forecast cone — {forward_days} fwd bars, n={n_samples} bootstrap",
        height=460,
    ))
    fig.update_yaxes(type="log", title_text="NAV (log)")
    grid_axes_update(fig)
    return fig


def forecast_cone_html(
    eq: pd.Series, label: str, div_id: str, forward_days: int = 252,
    n_samples: int = 5000,
) -> str:
    if len(eq) < 60:
        return f"<div id='{div_id}'>(insufficient data)</div>"
    fig = forecast_cone_figure(eq, label, forward_days, n_samples)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Perf-stats bootstrap box plots
# ---------------------------------------------------------------------------


def perf_stats_box_figure(
    eq: pd.Series, title: str, n_samples: int = 2000,
) -> go.Figure:
    fig = go.Figure()
    samples = perf_stats_bootstrap(eq, n_samples=n_samples)
    if samples.empty:
        fig.update_layout(**dark_layout(title + " (insufficient data)", height=420))
        return fig
    cols = ["sharpe", "sortino", "calmar", "omega", "cagr", "max_dd", "vol_ann"]
    palette = ["#58a6ff", "#3fb950", "#d29922", "#ce93d8", "#26a69a",
               "#ef5350", "#9ecbff"]
    for col, color in zip(cols, palette, strict=False):
        if col in samples.columns:
            fig.add_trace(go.Box(
                y=samples[col].values, name=col, marker_color=color,
                boxmean=True,
            ))
    fig.update_layout(**dark_layout(title, height=420))
    fig.update_yaxes(title_text="Bootstrap value")
    grid_axes_update(fig)
    return fig


def perf_stats_box_html(
    eq: pd.Series, title: str, div_id: str, n_samples: int = 2000,
) -> str:
    fig = perf_stats_box_figure(eq, title, n_samples)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def monte_carlo_paths_figure(
    eq: pd.Series, label: str, forward_days: int = 252,
    n_paths: int = 50, n_samples_for_terminal: int = 5000,
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.75, 0.25],
        subplot_titles=(
            f"{n_paths} sample paths, {forward_days} fwd bars",
            f"Terminal value (n={n_samples_for_terminal})",
        ),
        horizontal_spacing=0.06,
    )
    if len(eq) < 60:
        fig.update_layout(**dark_layout("Monte Carlo (insufficient data)", height=460))
        return fig
    rets = eq.pct_change().dropna().values
    start = float(eq.iloc[-1])
    rng = np.random.default_rng(0)

    sample_paths = rng.choice(rets, size=(n_paths, forward_days), replace=True)
    paths = start * (1 + sample_paths).cumprod(axis=1)

    terminal_paths = rng.choice(
        rets, size=(n_samples_for_terminal, forward_days), replace=True,
    )
    terminal_vals = start * (1 + terminal_paths).cumprod(axis=1)[:, -1]

    last_date = eq.index[-1]
    fwd_index = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=forward_days,
    )

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name=label,
        line=dict(color="#58a6ff", width=2), showlegend=False,
    ), row=1, col=1)
    for i in range(min(n_paths, 50)):
        fig.add_trace(go.Scatter(
            x=fwd_index, y=paths[i], mode="lines",
            line=dict(color="rgba(63,185,80,0.18)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Histogram(
        y=terminal_vals, marker_color="#d29922", opacity=0.85, nbinsy=40,
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(**dark_layout("Monte Carlo simulation", height=460))
    fig.update_yaxes(type="log", row=1, col=1, title_text="NAV (log)")
    grid_axes_update(fig)
    return fig


def monte_carlo_paths_html(
    eq: pd.Series, label: str, div_id: str, forward_days: int = 252,
    n_paths: int = 50, n_samples_for_terminal: int = 5000,
) -> str:
    if len(eq) < 60:
        return f"<div id='{div_id}'>(insufficient data)</div>"
    fig = monte_carlo_paths_figure(
        eq, label, forward_days, n_paths, n_samples_for_terminal,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
