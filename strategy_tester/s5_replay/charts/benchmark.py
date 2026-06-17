"""Benchmark-relative charts: regression scatter + cumulative excess +
rolling correlation + up/down capture + SPY-decile conditional returns.

Exposes both *_figure() and *_html() builders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)


# ---------------------------------------------------------------------------
# Regression scatter (strategy daily returns vs SPY + OLS)
# ---------------------------------------------------------------------------


def regression_scatter_figure(
    eq: pd.Series, bench_eq: pd.Series, label: str,
) -> go.Figure:
    fig = go.Figure()
    if bench_eq is None:
        fig.update_layout(**dark_layout("Regression (no benchmark)", height=460))
        return fig
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    rp = aligned["p"].pct_change().dropna() * 100
    rb = aligned["b"].pct_change().dropna() * 100
    rp, rb = rp.align(rb, join="inner")
    if len(rp) < 30 or float(rb.var()) <= 1e-12:
        fig.update_layout(**dark_layout("Regression (insufficient data)", height=460))
        return fig
    beta = float(rp.cov(rb) / rb.var())
    alpha = float(rp.mean() - beta * rb.mean())
    corr = float(rp.corr(rb))
    r2 = corr ** 2

    x_lo, x_hi = float(rb.min()), float(rb.max())
    pad = 0.05 * (x_hi - x_lo)
    x_line = np.linspace(x_lo - pad, x_hi + pad, 50)
    y_line = alpha + beta * x_line

    fig.add_trace(go.Scatter(
        x=rb.values, y=rp.values, mode="markers", name="Daily returns",
        marker=dict(size=4, color="#58a6ff", opacity=0.55),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        name=f"OLS  α={alpha:+.3f}  β={beta:.3f}  R²={r2:.3f}",
        line=dict(color="#ef5350", width=2.4),
    ))
    fig.add_vline(x=0, line=dict(color="#444", dash="dot", width=1))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(**dark_layout(
        f"{label} daily returns vs SPY", height=460,
    ))
    fig.update_xaxes(title_text="SPY return (%)")
    fig.update_yaxes(title_text=f"{label} return (%)")
    grid_axes_update(fig)
    return fig


def regression_scatter_html(
    eq: pd.Series, bench_eq: pd.Series, label: str, div_id: str,
) -> str:
    if bench_eq is None:
        return f"<div id='{div_id}'>(no benchmark)</div>"
    fig = regression_scatter_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Cumulative excess return (strat - bench) over time
# ---------------------------------------------------------------------------


def cumulative_excess_figure(
    eq: pd.Series, bench_eq: pd.Series, label: str,
) -> go.Figure:
    """Cum (strategy - benchmark) excess return. Positive slope = generating
    alpha; flat = matching market; negative = under-performing."""
    fig = go.Figure()
    if bench_eq is None or eq.empty:
        fig.update_layout(**dark_layout(
            "Cumulative excess return (no benchmark)", height=380,
        ))
        return fig
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    if len(aligned) < 30:
        fig.update_layout(**dark_layout(
            "Cumulative excess return (insufficient overlap)", height=380,
        ))
        return fig
    rp = aligned["p"].pct_change().fillna(0.0)
    rb = aligned["b"].pct_change().fillna(0.0)
    excess = rp - rb
    cum_excess = excess.cumsum() * 100  # in percentage points
    cum_p = ((1 + rp).cumprod() - 1) * 100
    cum_b = ((1 + rb).cumprod() - 1) * 100

    fig.add_trace(go.Scatter(
        x=cum_p.index, y=cum_p.values, mode="lines", name=label,
        line=dict(color="#58a6ff", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=cum_b.index, y=cum_b.values, mode="lines", name="SPY",
        line=dict(color="#ef5350", dash="dash", width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=cum_excess.index, y=cum_excess.values, mode="lines",
        name=f"{label} − SPY (excess, pp)",
        line=dict(color="#3fb950", width=2.2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.10)",
        yaxis="y2",
    ))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(
        **dark_layout(
            f"Cumulative return — {label} vs SPY (excess on right axis)",
            height=420,
        ),
        yaxis=dict(title=dict(text="Cum. return (%)"), gridcolor="#21262d"),
        yaxis2=dict(
            title=dict(text="Cum. excess (pp)", font=dict(color="#3fb950")),
            tickfont=dict(color="#3fb950"),
            overlaying="y", side="right", gridcolor="#21262d",
            showgrid=False,
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", zeroline=False)
    return fig


def cumulative_excess_html(
    eq: pd.Series, bench_eq: pd.Series, label: str, div_id: str,
) -> str:
    fig = cumulative_excess_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------


def rolling_correlation_figure(
    eq: pd.Series, bench_eq: pd.Series, label: str, window_days: int = 252,
) -> go.Figure:
    fig = go.Figure()
    if bench_eq is None:
        fig.update_layout(**dark_layout(
            "Rolling correlation vs SPY (no benchmark)", height=320,
        ))
        return fig
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    rp = aligned["p"].pct_change()
    rb = aligned["b"].pct_change()
    rolling_corr = rp.rolling(window_days).corr(rb).dropna()
    if rolling_corr.empty:
        fig.update_layout(**dark_layout(
            f"Rolling {window_days}-bar correlation (insufficient data)",
            height=320,
        ))
        return fig

    # Colored fill: above zero = co-moves with SPY, below = inverse
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values, mode="lines",
        name=f"{label} vs SPY",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
    ))
    fig.add_hline(
        y=float(rolling_corr.mean()),
        line=dict(color="#ffa726", dash="dash", width=1.4),
        annotation_text=f"mean ρ = {float(rolling_corr.mean()):.2f}",
        annotation_position="top right",
        annotation_font_color="#ffa726",
    )
    fig.add_hline(y=0, line=dict(color="#666", dash="solid", width=1))
    fig.add_hline(y=1, line=dict(color="#333", dash="dot", width=1))
    fig.add_hline(y=-1, line=dict(color="#333", dash="dot", width=1))
    fig.update_layout(**dark_layout(
        f"Rolling {window_days}-bar correlation vs SPY", height=340,
    ))
    fig.update_yaxes(title_text="Correlation ρ", range=[-1.05, 1.05])
    grid_axes_update(fig)
    return fig


def rolling_correlation_html(
    eq: pd.Series, bench_eq: pd.Series, label: str, div_id: str,
    window_days: int = 252,
) -> str:
    fig = rolling_correlation_figure(eq, bench_eq, label, window_days)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Up-capture vs down-capture
# ---------------------------------------------------------------------------


def up_down_capture_figure(
    eq: pd.Series, bench_eq: pd.Series, label: str,
) -> go.Figure:
    fig = go.Figure()
    if bench_eq is None:
        fig.update_layout(**dark_layout(
            "Up/Down capture (no benchmark)", height=380,
        ))
        return fig
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    rp = aligned["p"].pct_change().dropna()
    rb = aligned["b"].pct_change().dropna()
    rp, rb = rp.align(rb, join="inner")
    if len(rp) < 30:
        fig.update_layout(**dark_layout(
            "Up/Down capture (insufficient overlap)", height=380,
        ))
        return fig

    up_mask = rb > 0
    dn_mask = rb < 0
    up_cap = float(rp[up_mask].mean() / rb[up_mask].mean()) if up_mask.any() else np.nan
    dn_cap = float(rp[dn_mask].mean() / rb[dn_mask].mean()) if dn_mask.any() else np.nan
    asymmetry = up_cap - dn_cap if not (np.isnan(up_cap) or np.isnan(dn_cap)) else np.nan

    fig.add_trace(go.Bar(
        x=["Up-capture", "Down-capture"],
        y=[up_cap, dn_cap],
        marker_color=["#3fb950", "#ef5350"],
        text=[
            f"{up_cap:.2f}x" if not np.isnan(up_cap) else "—",
            f"{dn_cap:.2f}x" if not np.isnan(dn_cap) else "—",
        ],
        textposition="outside", textfont=dict(size=14, color="#e6edf3"),
        showlegend=False,
    ))
    fig.add_hline(
        y=1.0, line=dict(color="#9ecbff", dash="dash", width=1.5),
        annotation_text="1.0× (matches SPY)",
        annotation_position="bottom right",
        annotation_font_color="#9ecbff",
    )
    asym_str = f"{asymmetry:+.2f}" if not np.isnan(asymmetry) else "—"
    fig.update_layout(**dark_layout(
        f"Up vs Down capture (asymmetry = up − down = {asym_str})",
        height=400,
    ))
    fig.update_yaxes(title_text="Capture ratio", zeroline=True)
    grid_axes_update(fig)
    return fig


def up_down_capture_html(
    eq: pd.Series, bench_eq: pd.Series, label: str, div_id: str,
) -> str:
    fig = up_down_capture_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# SPY-decile conditional returns
# ---------------------------------------------------------------------------


def spy_decile_returns_figure(
    eq: pd.Series, bench_eq: pd.Series, label: str,
) -> go.Figure:
    """Bin SPY daily returns into 10 quantile buckets, show mean strategy
    return in each. Reveals whether the strategy's edge concentrates in
    up-SPY days (correlation alpha), down-SPY days (real diversifier), or
    is uniform (true alpha)."""
    fig = go.Figure()
    if bench_eq is None:
        fig.update_layout(**dark_layout(
            "SPY-decile conditional returns (no benchmark)", height=380,
        ))
        return fig
    aligned = pd.concat([eq, bench_eq], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["p", "b"])
    rp = aligned["p"].pct_change().dropna() * 100
    rb = aligned["b"].pct_change().dropna() * 100
    rp, rb = rp.align(rb, join="inner")
    if len(rp) < 100:
        fig.update_layout(**dark_layout(
            "SPY-decile conditional returns (insufficient data)", height=380,
        ))
        return fig

    df = pd.DataFrame({"p": rp.values, "b": rb.values})
    df["decile"] = pd.qcut(
        df["b"], 10, labels=False, duplicates="drop",
    )
    mean_by_decile = df.groupby("decile").agg(
        strat_mean=("p", "mean"),
        spy_mean=("b", "mean"),
        n=("p", "count"),
    ).reset_index()

    decile_labels = [
        f"D{int(d) + 1}\n({row['spy_mean']:+.2f}%)"
        for d, row in zip(mean_by_decile["decile"], mean_by_decile.to_dict("records"), strict=False)
    ]
    colors = [
        "#3fb950" if v >= 0 else "#ef5350"
        for v in mean_by_decile["strat_mean"]
    ]
    fig.add_trace(go.Bar(
        x=decile_labels, y=mean_by_decile["strat_mean"].values,
        marker_color=colors, name=label, showlegend=False,
        text=[f"{v:+.2f}%" for v in mean_by_decile["strat_mean"].values],
        textposition="outside", textfont=dict(size=10, color="#e6edf3"),
    ))
    fig.add_trace(go.Scatter(
        x=decile_labels, y=mean_by_decile["spy_mean"].values,
        mode="lines+markers", name="SPY mean (reference)",
        line=dict(color="#9ecbff", dash="dash", width=1.5),
        marker=dict(size=6),
    ))
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.update_layout(**dark_layout(
        f"Mean {label} return by SPY-return decile "
        f"(D1 = worst SPY days, D10 = best)",
        height=420,
    ))
    fig.update_yaxes(title_text="Mean daily return (%)")
    fig.update_xaxes(title_text="SPY decile (mean SPY return in parens)")
    grid_axes_update(fig)
    return fig


def spy_decile_returns_html(
    eq: pd.Series, bench_eq: pd.Series, label: str, div_id: str,
) -> str:
    fig = spy_decile_returns_figure(eq, bench_eq, label)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
