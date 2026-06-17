"""Underwater fill + top-N drawdown highlighted on equity + DD periods table.

Exposes both *_figure() and *_html() builders.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)


def underwater_figure(
    eq: pd.Series, spy_eq: pd.Series | None, label: str,
) -> go.Figure:
    fig = go.Figure()
    if eq.empty:
        fig.update_layout(**dark_layout("Underwater (no equity)", height=420))
        return fig
    dd = (eq / eq.cummax() - 1) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name=label,
        fill="tozeroy", line=dict(color="#ef5350", width=1.4),
        fillcolor="rgba(239,83,80,0.35)",
    ))
    if spy_eq is not None and not spy_eq.empty:
        spy_dd = (spy_eq / spy_eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=spy_dd.index, y=spy_dd.values, mode="lines", name="SPY",
            line=dict(color="#9ecbff", dash="dash", width=1.4),
        ))
    fig.update_layout(**dark_layout("Underwater plot — % below peak"))
    fig.update_yaxes(title_text="Drawdown (%)")
    grid_axes_update(fig)
    return fig


def underwater_html(
    eq: pd.Series, spy_eq: pd.Series | None, label: str, div_id: str,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = underwater_figure(eq, spy_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def gen_drawdown_table(eq: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Top-N peak→valley→recovery drawdown periods.

    Port of `pyfolio.timeseries.gen_drawdown_table`. recovery_date may be NaT
    for still-active drawdowns.
    """
    if len(eq) < 3:
        return pd.DataFrame()
    peak = eq.cummax()
    dd = eq / peak - 1
    in_dd = dd < 0
    if not in_dd.any():
        return pd.DataFrame()

    grp = (in_dd != in_dd.shift()).cumsum()
    episodes: list[dict] = []
    for _, idx in dd.groupby(grp).groups.items():
        sub = dd.loc[idx]
        if (sub < 0).all():
            valley_idx = sub.idxmin()
            valley_pos = eq.index.get_loc(valley_idx)
            peak_pos = valley_pos
            while peak_pos > 0 and eq.iloc[peak_pos - 1] >= eq.iloc[peak_pos]:
                peak_pos -= 1
            peak_date = eq.index[peak_pos]
            peak_value = eq.iloc[peak_pos]
            recovery_date = pd.NaT
            for after_pos in range(valley_pos + 1, len(eq)):
                if eq.iloc[after_pos] >= peak_value:
                    recovery_date = eq.index[after_pos]
                    break
            episodes.append({
                "peak_date": peak_date,
                "valley_date": valley_idx,
                "recovery_date": recovery_date,
                "dd_pct": float(sub.min()),
                "peak_to_valley_days": (valley_idx - peak_date).days,
                "valley_to_recovery_days": (
                    (recovery_date - valley_idx).days
                    if pd.notna(recovery_date) else None
                ),
                "duration_days": (
                    (recovery_date - peak_date).days
                    if pd.notna(recovery_date) else (eq.index[-1] - peak_date).days
                ),
            })
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame(episodes).sort_values("dd_pct").head(top_n).reset_index(drop=True)


def top_drawdowns_highlight_figure(
    eq: pd.Series, label: str, top_n: int = 5,
) -> go.Figure:
    fig = go.Figure()
    if eq.empty:
        fig.update_layout(**dark_layout("Top drawdowns (no equity)", height=420))
        return fig
    table = gen_drawdown_table(eq, top_n=top_n)
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name=label,
        line=dict(color="#58a6ff", width=2),
    ))
    for _, row in table.iterrows():
        end = (
            row["recovery_date"] if pd.notna(row["recovery_date"])
            else eq.index[-1]
        )
        fig.add_vrect(
            x0=row["peak_date"], x1=end,
            fillcolor="rgba(239,83,80,0.18)",
            line_width=0,
            annotation_text=f"{row['dd_pct'] * 100:.1f}%",
            annotation_position="top left",
            annotation=dict(font_color="#ef5350", font_size=10),
        )
    fig.update_layout(**dark_layout(
        f"Top {top_n} drawdowns highlighted on equity", height=420,
    ))
    fig.update_yaxes(type="log", title_text="NAV (log)")
    grid_axes_update(fig)
    return fig


def top_drawdowns_highlight_html(
    eq: pd.Series, label: str, div_id: str, top_n: int = 5,
) -> str:
    if eq.empty:
        return f"<div id='{div_id}'>(no equity)</div>"
    fig = top_drawdowns_highlight_figure(eq, label, top_n)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def drawdown_table_html(eq: pd.Series, top_n: int = 10) -> str:
    """HTML table of top-N drawdown periods for quant-ui-kit."""
    df = gen_drawdown_table(eq, top_n=top_n)
    if df.empty:
        return "<p>(no drawdown episodes)</p>"
    head = (
        "<thead><tr>"
        "<th data-col='rank' data-type='num' data-align='right'>#</th>"
        "<th data-col='dd' data-type='pct' data-align='right'>Max DD</th>"
        "<th data-col='peak' data-type='date' data-align='left'>Peak</th>"
        "<th data-col='valley' data-type='date' data-align='left'>Valley</th>"
        "<th data-col='recovery' data-type='date' data-align='left'>Recovery</th>"
        "<th data-col='p2v' data-type='num' data-align='right'>Days P→V</th>"
        "<th data-col='v2r' data-type='num' data-align='right'>Days V→R</th>"
        "<th data-col='total' data-type='num' data-align='right'>Total Days</th>"
        "</tr></thead>"
    )
    rows = []
    for i, row in df.iterrows():
        rec = (
            row["recovery_date"].strftime("%Y-%m-%d")
            if pd.notna(row["recovery_date"]) else "—"
        )
        v2r_raw = row["valley_to_recovery_days"]
        v2r = (
            f"{int(v2r_raw):,}"
            if v2r_raw is not None and pd.notna(v2r_raw) else "—"
        )
        rows.append(
            f"<tr>"
            f"<td>{int(i) + 1}</td>"
            f"<td data-tone='neg'>{row['dd_pct'] * 100:.2f}%</td>"
            f"<td class='date'>{row['peak_date'].strftime('%Y-%m-%d')}</td>"
            f"<td class='date'>{row['valley_date'].strftime('%Y-%m-%d')}</td>"
            f"<td class='date'>{rec}</td>"
            f"<td>{int(row['peak_to_valley_days']):,}</td>"
            f"<td>{v2r}</td>"
            f"<td>{int(row['duration_days']):,}</td>"
            f"</tr>"
        )
    return (
        "<div class='table-wrap'><div class='table-scroll'>"
        f"<table class='q-table sortable'>{head}<tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )
