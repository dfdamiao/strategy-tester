"""Round-trip trade analytics: lifetimes scatter, PnL distribution,
hold-time histogram, exit-reason breakdown, MAE/MFE scatter, per-symbol bar.

Consumes a DataFrame of closed trades with these columns (matches the
`TradeLog` dataclass):
    ticker, entry_date, exit_date, entry_price, exit_price, shares,
    pnl_dollars, pnl_pct, exit_reason, mae_pct, mfe_pct, hold_days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategy_tester.s5_replay.charts._common import (
    dark_layout,
    grid_axes_update,
)


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    """Coerce a list of TradeLog dataclass instances → DataFrame.

    Tolerates already-DataFrame input (passthrough). Empty list → empty DF
    with the canonical column schema.
    """
    if isinstance(trades, pd.DataFrame):
        return trades
    cols = [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "pnl_dollars", "pnl_pct", "exit_reason",
        "entry_pct_cash", "entry_pct_nav", "mae_pct", "mfe_pct", "hold_days",
        "commission_dollars",
    ]
    if not trades:
        return pd.DataFrame(columns=cols)
    rows = []
    for t in trades:
        rows.append({
            "ticker": getattr(t, "ticker", ""),
            "entry_date": pd.Timestamp(getattr(t, "entry_date", pd.NaT)),
            "exit_date": pd.Timestamp(getattr(t, "exit_date", pd.NaT)),
            "entry_price": float(getattr(t, "entry_price", float("nan"))),
            "exit_price": float(getattr(t, "exit_price", float("nan"))),
            "shares": int(getattr(t, "shares", 0)),
            "pnl_dollars": float(getattr(t, "pnl_dollars", float("nan"))),
            "pnl_pct": float(getattr(t, "pnl_pct", float("nan"))),
            "exit_reason": str(getattr(t, "exit_reason", "")),
            "entry_pct_cash": float(getattr(t, "entry_pct_cash", 0.0)),
            "entry_pct_nav": float(getattr(t, "entry_pct_nav", 0.0)),
            "mae_pct": float(getattr(t, "mae_pct", 0.0)),
            "mfe_pct": float(getattr(t, "mfe_pct", 0.0)),
            "hold_days": int(getattr(t, "hold_days", 0)),
            "commission_dollars": float(getattr(t, "commission_dollars", 0.0)),
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Round-trip lifetimes scatter
# ---------------------------------------------------------------------------


def round_trip_lifetimes_figure(trades_df: pd.DataFrame) -> go.Figure:
    """Scatter: x=entry_date, y=hold_days, color=pnl sign, size=abs(pnl_dollars).

    Pyfolio-style chart for "when did trades open and how long did they take?"
    """
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("Round-trip lifetimes (no trades)", height=400))
        return fig

    df = trades_df.copy()
    df["is_win"] = df["pnl_pct"] >= 0
    max_abs = float(df["pnl_dollars"].abs().max()) or 1.0
    df["size"] = (df["pnl_dollars"].abs() / max_abs * 22 + 4).clip(4, 26)

    for is_win, color, name in [
        (True, "#3fb950", "Winners"),
        (False, "#ef5350", "Losers"),
    ]:
        sub = df[df["is_win"] == is_win]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["entry_date"], y=sub["hold_days"],
            mode="markers",
            name=f"{name} (n={len(sub)})",
            marker=dict(
                color=color, size=sub["size"], opacity=0.6,
                line=dict(width=0.5, color="#0d1117"),
            ),
            customdata=np.stack([
                sub["ticker"], sub["pnl_pct"], sub["pnl_dollars"],
                sub["exit_reason"],
            ], axis=-1),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Entry: %{x|%Y-%m-%d}<br>"
                "Hold: %{y} days<br>"
                "PnL: %{customdata[1]:+.2f}% ($%{customdata[2]:+,.0f})<br>"
                "Exit: %{customdata[3]}<extra></extra>"
            ),
        ))
    fig.update_layout(**dark_layout(
        f"Round-trip lifetimes — {len(df)} closed trades", height=440,
    ))
    fig.update_xaxes(title_text="Entry date")
    fig.update_yaxes(title_text="Hold days")
    grid_axes_update(fig)
    return fig


def round_trip_lifetimes_html(
    trades_df: pd.DataFrame, div_id: str,
) -> str:
    fig = round_trip_lifetimes_figure(trades_df)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# PnL distribution histogram
# ---------------------------------------------------------------------------


def pnl_distribution_figure(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("PnL distribution (no trades)", height=360))
        return fig

    pnl = trades_df["pnl_pct"]
    fig.add_trace(go.Histogram(
        x=pnl, marker_color="#58a6ff", opacity=0.85, nbinsx=50,
        showlegend=False,
    ))
    mean = float(pnl.mean())
    med = float(pnl.median())
    fig.add_vline(
        x=0, line=dict(color="#666", dash="solid", width=1.2),
    )
    fig.add_vline(
        x=mean, line=dict(color="#ffa726", dash="dash", width=1.5),
        annotation_text=f"mean={mean:+.2f}%",
        annotation_position="top",
        annotation_font_color="#ffa726",
    )
    fig.add_vline(
        x=med, line=dict(color="#3fb950", dash="dot", width=1.5),
        annotation_text=f"median={med:+.2f}%",
        annotation_position="top right",
        annotation_font_color="#3fb950",
    )
    win_rate = float((pnl > 0).mean()) * 100
    fig.update_layout(**dark_layout(
        f"Per-trade PnL distribution — n={len(pnl)} · win rate {win_rate:.1f}%",
        height=380,
    ))
    fig.update_xaxes(title_text="PnL per trade (%)")
    fig.update_yaxes(title_text="Count")
    grid_axes_update(fig)
    return fig


def pnl_distribution_html(
    trades_df: pd.DataFrame, div_id: str,
) -> str:
    fig = pnl_distribution_figure(trades_df)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Hold-time distribution
# ---------------------------------------------------------------------------


def hold_time_distribution_figure(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("Hold-time distribution (no trades)", height=360))
        return fig

    hold = trades_df["hold_days"]
    # Split by outcome so the distribution shows win/loss durations separately
    wins = trades_df.loc[trades_df["pnl_pct"] >= 0, "hold_days"]
    losses = trades_df.loc[trades_df["pnl_pct"] < 0, "hold_days"]

    fig.add_trace(go.Histogram(
        x=wins, name=f"Wins (n={len(wins)})",
        marker_color="#3fb950", opacity=0.65, nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=losses, name=f"Losses (n={len(losses)})",
        marker_color="#ef5350", opacity=0.65, nbinsx=40,
    ))
    fig.update_layout(
        **dark_layout(
            f"Hold-time distribution — median {float(hold.median()):.0f}d, "
            f"mean {float(hold.mean()):.1f}d",
            height=380,
        ),
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Hold days")
    fig.update_yaxes(title_text="Count")
    grid_axes_update(fig)
    return fig


def hold_time_distribution_html(
    trades_df: pd.DataFrame, div_id: str,
) -> str:
    fig = hold_time_distribution_figure(trades_df)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Exit-reason breakdown (count + avg PnL)
# ---------------------------------------------------------------------------


def exit_reason_breakdown_figure(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("Exit reasons (no trades)", height=320))
        return fig

    grp = trades_df.groupby("exit_reason").agg(
        n=("pnl_pct", "size"),
        mean_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
        win_rate=("pnl_pct", lambda s: float((s > 0).mean()) * 100),
    ).reset_index().sort_values("n", ascending=False)

    fig.add_trace(go.Bar(
        x=grp["exit_reason"], y=grp["n"],
        marker_color=[
            "#58a6ff" if r == "signal"
            else "#ef5350" if r.startswith("stop")
            else "#d29922"
            for r in grp["exit_reason"]
        ],
        text=[
            f"{int(n)} · μ={m:+.2f}% · win {w:.0f}%"
            for n, m, w in zip(grp["n"], grp["mean_pnl"], grp["win_rate"], strict=False)
        ],
        textposition="outside", textfont=dict(size=11, color="#e6edf3"),
        showlegend=False,
    ))
    fig.update_layout(**dark_layout(
        f"Exit reason breakdown (n={int(grp['n'].sum())} total)", height=380,
    ))
    fig.update_xaxes(title_text="Exit reason")
    fig.update_yaxes(title_text="Trade count")
    grid_axes_update(fig)
    return fig


def exit_reason_breakdown_html(
    trades_df: pd.DataFrame, div_id: str,
) -> str:
    fig = exit_reason_breakdown_figure(trades_df)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# MAE/MFE scatter
# ---------------------------------------------------------------------------


def mae_mfe_scatter_figure(trades_df: pd.DataFrame) -> go.Figure:
    """Scatter: x=MAE% (worst unrealised drawdown during trade),
    y=MFE% (best unrealised gain), color=exit reason, marker=pnl_pct sign.

    Diagnostic: if MAE clusters far left and MFE clusters near zero,
    stops are too tight (capping winners) or trend filter too slow.
    """
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("MAE / MFE scatter (no trades)", height=420))
        return fig

    reasons = trades_df["exit_reason"].unique()
    color_map = {
        "signal": "#58a6ff", "stop_pct": "#ef5350",
        "stop_atr": "#d29922", "force_eow": "#ce93d8",
    }
    for r in reasons:
        sub = trades_df[trades_df["exit_reason"] == r]
        fig.add_trace(go.Scatter(
            x=sub["mae_pct"], y=sub["mfe_pct"], mode="markers",
            name=f"{r} (n={len(sub)})",
            marker=dict(
                color=color_map.get(r, "#888"),
                size=6, opacity=0.6,
                symbol=[
                    "circle" if p >= 0 else "x"
                    for p in sub["pnl_pct"]
                ],
            ),
            customdata=np.stack([
                sub["ticker"], sub["pnl_pct"], sub["hold_days"],
            ], axis=-1),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "MAE: %{x:+.2f}%<br>"
                "MFE: %{y:+.2f}%<br>"
                "PnL: %{customdata[1]:+.2f}% · Hold: %{customdata[2]}d"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.add_vline(x=0, line=dict(color="#444", dash="dot", width=1))
    # Diagonal: MFE = -MAE (where exit_pnl = avg(MAE, MFE) approximately)
    extent = max(
        float(trades_df["mfe_pct"].abs().max()),
        float(trades_df["mae_pct"].abs().max()),
    )
    fig.add_trace(go.Scatter(
        x=[-extent, extent], y=[-extent, extent],
        mode="lines", name="MFE = MAE",
        line=dict(color="#666", dash="dash", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(**dark_layout(
        f"MAE vs MFE (winners=circle, losers=×, n={len(trades_df)})",
        height=460,
    ))
    fig.update_xaxes(title_text="MAE % (worst unrealised excursion)")
    fig.update_yaxes(title_text="MFE % (best unrealised excursion)")
    grid_axes_update(fig)
    return fig


def mae_mfe_scatter_html(
    trades_df: pd.DataFrame, div_id: str,
) -> str:
    fig = mae_mfe_scatter_figure(trades_df)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Per-symbol PnL bar
# ---------------------------------------------------------------------------


def per_symbol_pnl_bar_figure(
    trades_df: pd.DataFrame, top_n: int = 25,
) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(**dark_layout("Per-symbol PnL (no trades)", height=420))
        return fig

    grp = trades_df.groupby("ticker").agg(
        total_pnl=("pnl_dollars", "sum"),
        n=("pnl_dollars", "size"),
        win_rate=("pnl_pct", lambda s: float((s > 0).mean()) * 100),
        avg_pnl_pct=("pnl_pct", "mean"),
    ).reset_index().sort_values("total_pnl", ascending=False)

    if len(grp) > top_n * 2:
        head = grp.head(top_n)
        tail = grp.tail(top_n)
        grp = pd.concat([head, tail])

    colors = ["#3fb950" if v >= 0 else "#ef5350" for v in grp["total_pnl"]]
    fig.add_trace(go.Bar(
        x=grp["ticker"], y=grp["total_pnl"], marker_color=colors,
        text=[
            f"n={int(n)} · win {w:.0f}% · μ {p:+.1f}%"
            for n, w, p in zip(grp["n"], grp["win_rate"], grp["avg_pnl_pct"], strict=False)
        ],
        textposition="outside", textfont=dict(size=9, color="#e6edf3"),
        showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.update_layout(**dark_layout(
        f"Per-symbol total PnL ($) — top + bottom {top_n} of {len(grp)} traded tickers",
        height=420,
    ))
    fig.update_xaxes(title_text="Ticker", tickangle=-45)
    fig.update_yaxes(title_text="Total PnL ($)")
    grid_axes_update(fig)
    return fig


def per_symbol_pnl_bar_html(
    trades_df: pd.DataFrame, div_id: str, top_n: int = 25,
) -> str:
    fig = per_symbol_pnl_bar_figure(trades_df, top_n)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Summary table for the quant-ui-kit
# ---------------------------------------------------------------------------


def trades_kpi_strip_html(
    trades_df: pd.DataFrame, seed_nav: float | None = None,
) -> str:
    """Top-of-Trades-tab KPI strip: n / win rate / gross+net PnL /
    **total commissions $ + % of NAV** / median hold / avg PnL%."""
    if trades_df.empty:
        return ""
    n = len(trades_df)
    gross_pnl = float(trades_df["pnl_dollars"].sum())
    commissions = float(trades_df["commission_dollars"].sum())
    net_pnl = gross_pnl - commissions
    win_rate = float((trades_df["pnl_pct"] > 0).mean()) * 100
    median_hold = float(trades_df["hold_days"].median())
    avg_pnl_pct = float(trades_df["pnl_pct"].mean())

    nav_note = (
        f"{commissions / seed_nav * 100:.2f}% of seed NAV"
        if seed_nav and seed_nav > 0 else ""
    )

    def card(label, value, tone, sub=""):
        return (
            f"<div class='kpi-card' data-tone='{tone}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value' data-tone='{tone}'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>"
        )

    return (
        "<div class='kpi-grid'>"
        + card("Closed trades", f"{n:,}", "accent",
               f"median hold {median_hold:.0f}d · avg PnL {avg_pnl_pct:+.2f}%")
        + card("Win rate", f"{win_rate:.1f}%",
               "pos" if win_rate >= 50 else "neg",
               f"{int((trades_df['pnl_pct'] > 0).sum()):,} wins / "
               f"{int((trades_df['pnl_pct'] <= 0).sum()):,} losses")
        + card("Gross PnL", f"${gross_pnl:+,.0f}",
               "pos" if gross_pnl > 0 else "neg",
               "before commissions")
        + card("Commissions", f"${commissions:,.0f}", "warn",
               nav_note)
        + card("Net PnL", f"${net_pnl:+,.0f}",
               "pos" if net_pnl > 0 else "neg",
               f"drag: ${commissions:,.0f} ({commissions / max(abs(gross_pnl), 1) * 100:.1f}% of gross)")
        + card("Best / worst trade",
               f"{float(trades_df['pnl_pct'].max()):+.1f}% / "
               f"{float(trades_df['pnl_pct'].min()):+.1f}%",
               "muted",
               f"MAE μ={float(trades_df['mae_pct'].mean()):.2f}% · "
               f"MFE μ={float(trades_df['mfe_pct'].mean()):.2f}%")
        + "</div>"
    )


def trades_full_table_html(
    trades_df: pd.DataFrame, max_rows: int | None = None,
) -> str:
    """Sortable + searchable table of every individual round trip.

    Quant-ui-kit auto-wires `.q-table.sortable` so column headers sort and
    the adjacent `.tt-search` input filters rows by ticker/exit-reason.
    `max_rows=None` shows everything; set to int to cap (rare — most
    strategies emit <2k trades).
    """
    if trades_df.empty:
        return "<p style='color:var(--muted)'>(no trade data)</p>"

    df = trades_df.copy()
    df = df.sort_values("entry_date", ascending=True).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    head = (
        "<thead><tr>"
        "<th data-col='n' data-type='num' data-align='right'>#</th>"
        "<th data-col='ticker' data-type='text' data-align='left'>Ticker</th>"
        "<th data-col='entry' data-type='date' data-align='left'>Entry</th>"
        "<th data-col='exit' data-type='date' data-align='left'>Exit</th>"
        "<th data-col='hold' data-type='num' data-align='right'>Hold (d)</th>"
        "<th data-col='entry_px' data-type='money' data-align='right'>Entry $</th>"
        "<th data-col='exit_px' data-type='money' data-align='right'>Exit $</th>"
        "<th data-col='shares' data-type='num' data-align='right'>Shares</th>"
        "<th data-col='pnl_pct' data-type='num' data-align='right'>PnL %</th>"
        "<th data-col='pnl_$' data-type='money' data-align='right'>PnL $</th>"
        "<th data-col='mae' data-type='num' data-align='right'>MAE %</th>"
        "<th data-col='mfe' data-type='num' data-align='right'>MFE %</th>"
        "<th data-col='entry_pct_nav' data-type='num' data-align='right'>Entry % NAV</th>"
        "<th data-col='comm' data-type='money' data-align='right'>Comm $</th>"
        "<th data-col='reason' data-type='text' data-align='left'>Exit reason</th>"
        "</tr></thead>"
    )

    rows = []
    for i, r in df.iterrows():
        tone_pnl = "pos" if r["pnl_pct"] >= 0 else "neg"
        rows.append(
            f"<tr data-result='{'win' if r['pnl_pct'] >= 0 else 'loss'}'>"
            f"<td>{int(i) + 1}</td>"
            f"<td><b>{r['ticker']}</b></td>"
            f"<td class='date'>{pd.Timestamp(r['entry_date']).strftime('%Y-%m-%d')}</td>"
            f"<td class='date'>{pd.Timestamp(r['exit_date']).strftime('%Y-%m-%d')}</td>"
            f"<td>{int(r['hold_days']):,}</td>"
            f"<td>${r['entry_price']:,.2f}</td>"
            f"<td>${r['exit_price']:,.2f}</td>"
            f"<td>{int(r['shares']):,}</td>"
            f"<td data-tone='{tone_pnl}'>{r['pnl_pct']:+.2f}%</td>"
            f"<td data-tone='{tone_pnl}'>${r['pnl_dollars']:+,.0f}</td>"
            f"<td data-tone='neg'>{r['mae_pct']:+.2f}%</td>"
            f"<td data-tone='pos'>{r['mfe_pct']:+.2f}%</td>"
            f"<td>{r['entry_pct_nav']:.2f}%</td>"
            f"<td data-tone='warn'>${r['commission_dollars']:,.2f}</td>"
            f"<td>{r['exit_reason']}</td>"
            f"</tr>"
        )

    total_pnl = float(df["pnl_dollars"].sum())
    total_comm = float(df["commission_dollars"].sum())
    win_rate = float((df["pnl_pct"] > 0).mean()) * 100
    totals_row = (
        "<tr class='totals'>"
        "<td>TOTAL</td>"
        f"<td>{len(df)} trades</td>"
        "<td>—</td><td>—</td>"
        f"<td>{float(df['hold_days'].mean()):.1f}μ</td>"
        "<td>—</td><td>—</td>"
        f"<td>{int(df['shares'].sum()):,}</td>"
        f"<td>{float(df['pnl_pct'].mean()):+.2f}%μ</td>"
        f"<td data-tone='{'pos' if total_pnl > 0 else 'neg'}'>${total_pnl:+,.0f}</td>"
        f"<td data-tone='neg'>{float(df['mae_pct'].mean()):+.2f}%μ</td>"
        f"<td data-tone='pos'>{float(df['mfe_pct'].mean()):+.2f}%μ</td>"
        f"<td>{float(df['entry_pct_nav'].mean()):.2f}%μ</td>"
        f"<td data-tone='warn'>${total_comm:,.0f}</td>"
        f"<td>win {win_rate:.1f}%</td>"
        "</tr>"
    )

    toolbar = (
        "<div class='table-toolbar'>"
        "<div><span class='tt-label'>All trades</span>"
        f"<span class='tt-meta' style='margin-left:10px'>"
        f"{len(df)} rows · sortable · filter by ticker / exit reason</span>"
        "</div>"
        "<input type='text' class='tt-search' placeholder='Filter…'>"
        "</div>"
    )
    return (
        "<div class='table-wrap'>"
        f"{toolbar}"
        "<div class='table-scroll' style='max-height:560px;overflow-y:auto'>"
        f"<table class='q-table sortable'>{head}<tbody>"
        + "".join(rows)
        + totals_row
        + "</tbody></table></div></div>"
    )


def trades_summary_table_html(trades_df: pd.DataFrame) -> str:
    """One-row-per-exit-reason summary stats + total-commissions column."""
    if trades_df.empty:
        return "<p style='color:var(--muted)'>(no trade data — strategy may not have populated state.trades)</p>"

    head = (
        "<thead><tr>"
        "<th data-col='reason' data-type='text' data-align='left'>Exit</th>"
        "<th data-col='n' data-type='num' data-align='right'>Trades</th>"
        "<th data-col='wr' data-type='pct' data-align='right'>Win rate</th>"
        "<th data-col='mean' data-type='num' data-align='right'>Mean PnL</th>"
        "<th data-col='median' data-type='num' data-align='right'>Median PnL</th>"
        "<th data-col='best' data-type='num' data-align='right'>Best</th>"
        "<th data-col='worst' data-type='num' data-align='right'>Worst</th>"
        "<th data-col='hold' data-type='num' data-align='right'>Median hold</th>"
        "<th data-col='mae' data-type='num' data-align='right'>Mean MAE</th>"
        "<th data-col='mfe' data-type='num' data-align='right'>Mean MFE</th>"
        "<th data-col='comm' data-type='money' data-align='right'>Σ comm $</th>"
        "</tr></thead>"
    )
    rows = []
    grp = trades_df.groupby("exit_reason")
    overall_row = {
        "reason": f"<b>ALL ({len(trades_df)})</b>",
        "n": len(trades_df),
        "wr": float((trades_df["pnl_pct"] > 0).mean()) * 100,
        "mean": float(trades_df["pnl_pct"].mean()),
        "median": float(trades_df["pnl_pct"].median()),
        "best": float(trades_df["pnl_pct"].max()),
        "worst": float(trades_df["pnl_pct"].min()),
        "hold": float(trades_df["hold_days"].median()),
        "mae": float(trades_df["mae_pct"].mean()),
        "mfe": float(trades_df["mfe_pct"].mean()),
        "comm": float(trades_df["commission_dollars"].sum()),
    }

    def _row(r):
        tone_mean = "pos" if r["mean"] > 0 else "neg"
        return (
            f"<tr>"
            f"<td>{r['reason']}</td>"
            f"<td>{int(r['n']):,}</td>"
            f"<td>{r['wr']:.1f}%</td>"
            f"<td data-tone='{tone_mean}'>{r['mean']:+.2f}%</td>"
            f"<td>{r['median']:+.2f}%</td>"
            f"<td data-tone='pos'>{r['best']:+.2f}%</td>"
            f"<td data-tone='neg'>{r['worst']:+.2f}%</td>"
            f"<td>{r['hold']:.0f}d</td>"
            f"<td data-tone='neg'>{r['mae']:+.2f}%</td>"
            f"<td data-tone='pos'>{r['mfe']:+.2f}%</td>"
            f"<td data-tone='warn'>${r['comm']:,.0f}</td>"
            f"</tr>"
        )

    rows.append(_row(overall_row))
    for reason, sub in grp:
        rows.append(_row({
            "reason": reason,
            "n": len(sub),
            "wr": float((sub["pnl_pct"] > 0).mean()) * 100,
            "mean": float(sub["pnl_pct"].mean()),
            "median": float(sub["pnl_pct"].median()),
            "best": float(sub["pnl_pct"].max()),
            "worst": float(sub["pnl_pct"].min()),
            "hold": float(sub["hold_days"].median()),
            "mae": float(sub["mae_pct"].mean()),
            "mfe": float(sub["mfe_pct"].mean()),
            "comm": float(sub["commission_dollars"].sum()),
        }))
    return (
        "<div class='table-wrap'><div class='table-scroll'>"
        f"<table class='q-table sortable'>{head}<tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )
