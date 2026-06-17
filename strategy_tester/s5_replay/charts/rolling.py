"""Rolling Sharpe / Sortino / Vol / Beta time series charts.

Exposes both *_figure() and *_html() builders.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)
from strategy_tester.s5_replay.extra_metrics import (
    rolling_beta,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
)


def _two_line_figure(
    s1: pd.Series, s1_name: str,
    s2: pd.Series | None, s2_name: str,
    title: str, y_title: str,
    add_zero_line: bool = True,
) -> go.Figure:
    fig = go.Figure()
    if not s1.empty:
        fig.add_trace(go.Scatter(
            x=s1.index, y=s1.values, mode="lines", name=s1_name,
            line=dict(color="#58a6ff", width=2),
        ))
    if s2 is not None and not s2.empty:
        fig.add_trace(go.Scatter(
            x=s2.index, y=s2.values, mode="lines", name=s2_name,
            line=dict(color="#ef5350", dash="dash", width=1.5),
        ))
    if add_zero_line:
        fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(**dark_layout(title, height=320))
    fig.update_yaxes(title_text=y_title)
    grid_axes_update(fig)
    return fig


# ---------------------------------------------------------------------------


def rolling_sharpe_figure(
    eq: pd.Series, spy_eq: pd.Series | None, label: str,
    window_days: int = 252,
) -> go.Figure:
    s = rolling_sharpe(eq, window_days)
    b = rolling_sharpe(spy_eq, window_days) if spy_eq is not None else None
    return _two_line_figure(
        s, label, b, "SPY",
        f"Rolling Sharpe — {window_days}-bar window", "Sharpe",
    )


def rolling_sharpe_html(
    eq: pd.Series, spy_eq: pd.Series | None, label: str, div_id: str,
    window_days: int = 252,
) -> str:
    fig = rolling_sharpe_figure(eq, spy_eq, label, window_days)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def rolling_sortino_figure(
    eq: pd.Series, spy_eq: pd.Series | None, label: str,
    window_days: int = 252,
) -> go.Figure:
    s = rolling_sortino(eq, window_days)
    b = rolling_sortino(spy_eq, window_days) if spy_eq is not None else None
    return _two_line_figure(
        s, label, b, "SPY",
        f"Rolling Sortino — {window_days}-bar window", "Sortino",
    )


def rolling_sortino_html(
    eq: pd.Series, spy_eq: pd.Series | None, label: str, div_id: str,
    window_days: int = 252,
) -> str:
    fig = rolling_sortino_figure(eq, spy_eq, label, window_days)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def rolling_volatility_figure(
    eq: pd.Series, spy_eq: pd.Series | None, label: str,
    window_days: int = 252,
) -> go.Figure:
    s = rolling_volatility(eq, window_days) * 100
    b = (
        rolling_volatility(spy_eq, window_days) * 100
        if spy_eq is not None else None
    )
    return _two_line_figure(
        s, label, b, "SPY",
        f"Rolling Volatility — {window_days}-bar window",
        "Vol (% ann)", add_zero_line=False,
    )


def rolling_volatility_html(
    eq: pd.Series, spy_eq: pd.Series | None, label: str, div_id: str,
    window_days: int = 252,
) -> str:
    fig = rolling_volatility_figure(eq, spy_eq, label, window_days)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def rolling_beta_figure(
    eq: pd.Series, spy_eq: pd.Series | None, label: str,
    window_days: int = 252,
) -> go.Figure:
    fig = go.Figure()
    if spy_eq is None:
        fig.update_layout(**dark_layout(
            f"Rolling Beta vs SPY — {window_days}-bar window (no SPY)",
            height=320,
        ))
        return fig
    s = rolling_beta(eq, spy_eq, window_days)
    if not s.empty:
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(color="#58a6ff", width=2),
        ))
    fig.add_hline(y=1, line=dict(color="#ef5350", dash="dash", width=1))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(**dark_layout(
        f"Rolling Beta vs SPY — {window_days}-bar window", height=320,
    ))
    fig.update_yaxes(title_text="Beta")
    grid_axes_update(fig)
    return fig


def rolling_beta_html(
    eq: pd.Series, spy_eq: pd.Series, label: str, div_id: str,
    window_days: int = 252,
) -> str:
    if spy_eq is None:
        return f"<div id='{div_id}'>(no SPY benchmark)</div>"
    fig = rolling_beta_figure(eq, spy_eq, label, window_days)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
