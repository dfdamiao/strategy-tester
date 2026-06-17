"""Interesting-times small-multiples grid (pyfolio-style stress windows).

Exposes both *_figure() and *_html() builders.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)
from strategy_tester.s5_replay.stress_windows import (
    WINDOWS,
    slice_equity,
    window_stats,
)


def stress_grid_figure(
    eq: pd.Series, bench_eq: pd.Series | None, label: str, cols: int = 5,
) -> go.Figure:
    """Small-multiples grid: one cum-return chart per named stress window."""
    valid: list[tuple[str, pd.Series, pd.Series | None]] = []
    for name, (start, end) in WINDOWS.items():
        s = slice_equity(eq, start, end)
        if len(s) < 5:
            continue
        b = slice_equity(bench_eq, start, end) if bench_eq is not None else None
        valid.append((name, s, b))

    fig = go.Figure()
    if not valid:
        fig.update_layout(**dark_layout(
            "Stress windows (none overlap)", height=320,
        ))
        return fig

    rows = (len(valid) + cols - 1) // cols
    titles = [v[0].replace("_", " ") for v in valid]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.08, horizontal_spacing=0.04,
    )
    for i, (_, s, b) in enumerate(valid):
        r, c = i // cols + 1, i % cols + 1
        cum = s / s.iloc[0] - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100, mode="lines",
            line=dict(color="#58a6ff", width=1.5),
            showlegend=(i == 0), name=label,
        ), row=r, col=c)
        if b is not None and len(b) >= 5:
            b_cum = b / b.iloc[0] - 1
            fig.add_trace(go.Scatter(
                x=b_cum.index, y=b_cum.values * 100, mode="lines",
                line=dict(color="#ef5350", dash="dash", width=1.2),
                showlegend=(i == 0), name="SPY",
            ), row=r, col=c)
        fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=0.5),
                      row=r, col=c)

    height = max(260 * rows, 320)
    fig.update_layout(**dark_layout(
        f"Stress windows — {len(valid)} of {len(WINDOWS)} overlap equity range",
        height=height,
    ))
    fig.update_annotations(font_size=10)
    grid_axes_update(fig)
    return fig


def stress_grid_html(
    eq: pd.Series, bench_eq: pd.Series | None, label: str, div_id: str,
    cols: int = 5,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = stress_grid_figure(eq, bench_eq, label, cols)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def stress_table_html(eq: pd.Series, bench_eq: pd.Series | None) -> str:
    """Per-window cum-return / Sharpe / MaxDD table styled for quant-ui-kit."""
    df = window_stats(eq, bench_eq)
    if df.empty:
        return "<p>(no overlapping stress windows)</p>"
    head = (
        "<thead><tr>"
        "<th data-col='window' data-type='text' data-align='left'>Window</th>"
        "<th data-col='start' data-type='date' data-align='left'>Start</th>"
        "<th data-col='end' data-type='date' data-align='left'>End</th>"
        "<th data-col='bars' data-type='num' data-align='right'>Bars</th>"
        "<th data-col='ret' data-type='pct' data-align='right'>Cum Ret</th>"
        "<th data-col='bench' data-type='pct' data-align='right'>SPY Ret</th>"
        "<th data-col='excess' data-type='pct' data-align='right'>Excess</th>"
        "<th data-col='sharpe' data-type='num' data-align='right'>Sharpe</th>"
        "<th data-col='dd' data-type='pct' data-align='right'>Max DD</th>"
        "</tr></thead>"
    )
    rows = []
    for _, r in df.iterrows():
        tone_excess = (
            "pos" if pd.notna(r["excess_return"]) and r["excess_return"] > 0
            else "neg" if pd.notna(r["excess_return"]) and r["excess_return"] < 0
            else "muted"
        )
        bench_cell = (
            f"{r['bench_return'] * 100:+.2f}%"
            if pd.notna(r["bench_return"]) else "—"
        )
        excess_cell = (
            f"{r['excess_return'] * 100:+.2f}%"
            if pd.notna(r["excess_return"]) else "—"
        )
        sharpe_cell = (
            f"{r['sharpe']:.2f}" if pd.notna(r["sharpe"]) else "—"
        )
        rows.append(
            f"<tr>"
            f"<td>{r['window'].replace('_', ' ')}</td>"
            f"<td class='date'>{r['start']}</td>"
            f"<td class='date'>{r['end']}</td>"
            f"<td>{int(r['n_bars']):,}</td>"
            f"<td data-tone='{'pos' if r['cum_return'] > 0 else 'neg'}'>"
            f"{r['cum_return'] * 100:+.2f}%</td>"
            f"<td>{bench_cell}</td>"
            f"<td data-tone='{tone_excess}'>{excess_cell}</td>"
            f"<td>{sharpe_cell}</td>"
            f"<td data-tone='neg'>{r['max_dd'] * 100:+.2f}%</td>"
            f"</tr>"
        )
    return (
        "<div class='table-wrap'><div class='table-scroll'>"
        f"<table class='q-table sortable'>{head}<tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )
