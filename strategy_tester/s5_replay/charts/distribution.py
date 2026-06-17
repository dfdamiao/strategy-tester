"""Return-distribution charts: histograms, boxplots, KDE overlay vs Normal.

Exposes both *_figure() for the picker and *_html() for direct embed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)


def _resample_returns(eq: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return eq.pct_change().dropna()
    return eq.resample(freq).last().pct_change().dropna()


# ---------------------------------------------------------------------------
# 3-panel histogram (D / W / M)
# ---------------------------------------------------------------------------


def returns_histogram_figure(eq: pd.Series, title: str) -> go.Figure:
    daily = _resample_returns(eq, "D") * 100
    weekly = _resample_returns(eq, "W") * 100
    monthly = _resample_returns(eq, "ME") * 100

    def _title(label: str, s: pd.Series) -> str:
        if s.empty:
            return label
        return f"{label} · μ={float(s.mean()):+.2f}  σ={float(s.std(ddof=1)):.2f}"

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            _title("Daily (%)", daily),
            _title("Weekly (%)", weekly),
            _title("Monthly (%)", monthly),
        ),
        horizontal_spacing=0.06,
    )
    for col, s, color in [
        (1, daily, "#58a6ff"),
        (2, weekly, "#3fb950"),
        (3, monthly, "#d29922"),
    ]:
        if not s.empty:
            fig.add_trace(go.Histogram(
                x=s.values, name="", marker_color=color, opacity=0.85,
                nbinsx=40, showlegend=False,
            ), row=1, col=col)
            mean = float(s.mean())
            fig.add_vline(
                x=mean, line=dict(color="#ffa726", dash="dash", width=1.5),
                row=1, col=col,
            )

    fig.update_layout(**dark_layout(title, height=340))
    fig.update_annotations(font_size=11)
    grid_axes_update(fig)
    return fig


def returns_histogram_html(
    eq: pd.Series, title: str, div_id: str,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = returns_histogram_figure(eq, title)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Boxplots D / W / M
# ---------------------------------------------------------------------------


def returns_boxplot_figure(eq: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if eq.empty:
        fig.update_layout(**dark_layout(title, height=380))
        return fig
    daily = _resample_returns(eq, "D") * 100
    weekly = _resample_returns(eq, "W") * 100
    monthly = _resample_returns(eq, "ME") * 100
    for name, s, color in [
        ("Daily", daily, "#58a6ff"),
        ("Weekly", weekly, "#3fb950"),
        ("Monthly", monthly, "#d29922"),
    ]:
        if not s.empty:
            fig.add_trace(go.Box(
                y=s.values, name=name, marker_color=color, boxmean=True,
            ))
    fig.update_layout(**dark_layout(title, height=380))
    fig.update_yaxes(title_text="Return (%)")
    grid_axes_update(fig)
    return fig


def returns_boxplot_html(
    eq: pd.Series, title: str, div_id: str,
) -> str:
    fig = returns_boxplot_figure(eq, title)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# KDE vs Normal
# ---------------------------------------------------------------------------


def kde_vs_normal_figure(eq: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    rets = eq.pct_change().dropna().values * 100
    if len(rets) < 30:
        fig.update_layout(**dark_layout(title + " (insufficient)", height=360))
        return fig
    kde = gaussian_kde(rets)
    x = np.linspace(rets.min() - 1, rets.max() + 1, 400)
    y_kde = kde(x)
    mu, sigma = float(np.mean(rets)), float(np.std(rets, ddof=1))
    y_norm = norm.pdf(x, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x, y=y_kde, mode="lines", name="Empirical KDE",
        line=dict(color="#58a6ff", width=2.2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_norm, mode="lines",
        name=f"Normal(μ={mu:.2f}, σ={sigma:.2f})",
        line=dict(color="#ef5350", dash="dash", width=1.6),
    ))
    fig.update_layout(**dark_layout(title, height=360))
    fig.update_yaxes(title_text="Density")
    fig.update_xaxes(title_text="Daily return (%)")
    grid_axes_update(fig)
    return fig


def kde_vs_normal_html(eq: pd.Series, title: str, div_id: str) -> str:
    fig = kde_vs_normal_figure(eq, title)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
