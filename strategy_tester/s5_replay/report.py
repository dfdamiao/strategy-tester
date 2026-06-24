"""HTML report + Plotly overlays.

Extracted verbatim from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` lines
~969-1769 (2026-04-30). Strategy-agnostic (the `vs_old_s5_html` block
that compares against hardcoded paleologo_verification.md numbers stays in
the strategy adapter; this lib accepts it as an opaque HTML fragment).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy_tester.s5_replay.metrics import spy_summary_row
from strategy_tester.s5_replay.oracles import scheme_per_name_cap
from strategy_tester.s5_replay.runner import SchemeResult
from strategy_tester.s5_replay.walker import DEFAULT_PER_NAME_CAP

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def fmt_pct(x: float, sign: bool = True) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def fmt_num(x: float, dp: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.{dp}f}"


def fmt_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


_PCT_DEC = {
    "oos_cagr",
    "oos_max_dd",
    "oos_vol_ann",
    "full_cagr",
    "full_max_dd",
    "full_vol_ann",
    "alpha_ann",
    "tracking_error",
    "excess_cagr",
}
_PCT_RAW = {
    "oos_hit_rate",
    "oos_pct_invested",
    "full_hit_rate",
    "full_pct_invested",
    "mean_clip_ratio",
    "up_capture",
    "down_capture",
}
_NUM3 = {
    "oos_sharpe",
    "oos_calmar",
    "oos_sortino",
    "oos_martin",
    "full_sharpe",
    "full_calmar",
    "full_sortino",
    "full_martin",
    "ir_vs_spy",
    "beta",
    "corr_spy",
}
_NUM2 = {
    "oos_ulcer",
    "oos_skew",
    "oos_kurtosis",
    "full_ulcer",
    "full_skew",
    "full_kurtosis",
    "max_entry_pct_cash",
    "p95_entry_pct_cash",
    "max_entry_pct_nav",
}
_INT = {
    "n_trades",
    "n_failed_entries",
    "n_clipped",
    "n_too_small",
    "oos_dd_dur_days",
    "full_dd_dur_days",
}
_MONEY = {"end_value", "oos_end_value"}
# Columns that compare a scheme to the SPY benchmark — colored red in HTML
# so they stand out against the scheme's own metrics.
_SPY_COLS = {
    "ir_vs_spy",
    "alpha_ann",
    "beta",
    "corr_spy",
    "tracking_error",
    "up_capture",
    "down_capture",
    "excess_cagr",
}


def _fmt_cell(col: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if col in _PCT_DEC and isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v) * 100:+.2f}%"
    if col in _PCT_RAW and isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v) * 100:.1f}%"
    if col in _NUM3 and isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.3f}"
    if col in _NUM2 and isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.2f}"
    if col in _INT and isinstance(v, (int, float, np.integer, np.floating)):
        return f"{int(v):,}"
    if col in _MONEY and isinstance(v, (int, float, np.integer, np.floating)):
        return f"${float(v):,.0f}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _sort_val(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-1e9"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(float(v))
    return str(v)


def summary_table_html(
    summary: pd.DataFrame,
    top3: set[str],
    spy_row: dict | None = None,
) -> str:
    cols = list(summary.columns)
    head = "".join(
        f"<th data-col='{c}' class='{('cell-spy' if c in _SPY_COLS else '')}'>"
        f"{c.replace('_', ' ')}</th>"
        for c in cols
    )
    body_rows = []
    # Pinned SPY benchmark row (if available) — sticks to the top so users
    # always have a benchmark visible regardless of column sort order.
    if spy_row is not None:
        cells = []
        for col in cols:
            v = spy_row.get(col)
            cells.append(
                f"<td data-val='{_sort_val(v)}' data-col='{col}'>"
                f"{_fmt_cell(col, v)}</td>"
            )
        body_rows.append(
            "<tr class='row-spy' data-pinned='1'>" + "".join(cells) + "</tr>"
        )
    for _, r in summary.iterrows():
        cls = "row-top" if r["scheme"] in top3 else ""
        cells = []
        for col in cols:
            v = r[col]
            cell_cls = "cell-spy" if col in _SPY_COLS else ""
            cells.append(
                f"<td data-val='{_sort_val(v)}' data-col='{col}' "
                f"class='{cell_cls}'>{_fmt_cell(col, v)}</td>"
            )
        body_rows.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")
    body = "\n".join(body_rows)
    return f"""
<table id='nrtbl' class='summary-table'>
  <thead><tr>{head}</tr></thead>
  <tbody>
{body}
  </tbody>
</table>
<script>
(function() {{
  const tbl = document.getElementById('nrtbl');
  const thead = tbl.querySelector('thead');
  const tbody = tbl.querySelector('tbody');
  let sortDir = {{}};
  thead.querySelectorAll('th').forEach(th => {{
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {{
      const col = th.dataset.col;
      const asc = !sortDir[col];
      sortDir = {{ [col]: asc }};
      const allRows = Array.from(tbody.querySelectorAll('tr'));
      const pinned = allRows.filter(r => r.dataset.pinned === '1');
      const sortable = allRows.filter(r => r.dataset.pinned !== '1');
      sortable.sort((a, b) => {{
        const av = a.querySelector(`td[data-col='${{col}}']`).dataset.val;
        const bv = b.querySelector(`td[data-col='${{col}}']`).dataset.val;
        const an = parseFloat(av);
        const bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});
      pinned.forEach(r => tbody.appendChild(r));
      sortable.forEach(r => tbody.appendChild(r));
    }});
  }});
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Equity / drawdown / cash mobilization overlays
# ---------------------------------------------------------------------------


def _slice_rebase(
    eq: pd.Series,
    start: pd.Timestamp | None,
    base: float | None,
) -> pd.Series:
    """Slice ``eq`` at ``start`` (inclusive) and rebase first value to
    ``base``. Pass ``start=None`` for the full series and ``base=None`` to
    keep absolute scale. Used by the OOS-only overlays so portfolio + SPY
    + buy-hold all start at the same NetLiq for a fair visual."""
    out = eq if start is None else eq.loc[eq.index >= start]
    if base is None or out.empty or out.iloc[0] == 0:
        return out
    return out * (base / float(out.iloc[0]))


def equity_overlay(
    results: list[SchemeResult],
    top_schemes: list[str],
    spy_eq: pd.Series | None,
    bh_eq: pd.Series,
    oos_start: pd.Timestamp,
    title: str,
    *,
    bh_label: str = "buy-hold",
    spy_label: str = "SPY",
    window_start: pd.Timestamp | None = None,
    rebase_to: float | None = None,
    div_id: str = "fig-eq",
) -> str:
    """Plot scheme equity + benchmark + buy-hold. If ``window_start``
    is given, slice all series at that date and rebase the first value to
    ``rebase_to`` so the OOS-only and full-period panels are directly
    comparable."""
    fig = go.Figure()
    by_name = {r.scheme: r for r in results}
    for i, scheme in enumerate(top_schemes):
        if scheme not in by_name:
            continue
        eq = _slice_rebase(by_name[scheme].equity, window_start, rebase_to)
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.values,
                mode="lines",
                name=scheme,
                line=dict(color=color, width=2),
            )
        )
    if spy_eq is not None:
        spy = _slice_rebase(spy_eq, window_start, rebase_to)
        fig.add_trace(
            go.Scatter(
                x=spy.index,
                y=spy.values,
                mode="lines",
                name=spy_label,
                line=dict(color="#ef5350", dash="dash", width=2),
            )
        )
    bh = _slice_rebase(bh_eq, window_start, rebase_to)
    fig.add_trace(
        go.Scatter(
            x=bh.index,
            y=bh.values,
            mode="lines",
            name=bh_label,
            line=dict(color="#bbbbbb", dash="dot", width=1.5),
        )
    )
    if window_start is None:
        fig.add_vrect(
            x0=oos_start,
            x1=max(r.equity.index.max() for r in results),
            fillcolor="#26a69a",
            opacity=0.08,
            line_width=0,
            annotation_text="OOS ->",
            annotation_position="top left",
            annotation=dict(font_color="#26a69a"),
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#181818",
        margin=dict(l=50, r=180, t=50, b=40),
        height=520,
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            bgcolor="rgba(26,26,26,0.8)",
            font=dict(size=10),
        ),
        yaxis=dict(type="log", title="NetLiq (log)"),
    )
    fig.update_xaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a")
    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=div_id)


def drawdown_overlay(
    results: list[SchemeResult],
    top_schemes: list[str],
    spy_eq: pd.Series | None,
    title: str,
    *,
    window_start: pd.Timestamp | None = None,
    spy_label: str = "SPY",
    div_id: str = "fig-dd",
) -> str:
    """Plot drawdown for each scheme + SPY. ``window_start`` slices all
    series before computing the cummax (so OOS-only DD is the realised DD
    *within* the OOS window, not carried over from before)."""
    fig = go.Figure()
    by_name = {r.scheme: r for r in results}
    for i, scheme in enumerate(top_schemes):
        if scheme not in by_name:
            continue
        eq = _slice_rebase(by_name[scheme].equity, window_start, None)
        dd = (eq / eq.cummax() - 1) * 100
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=scheme,
                line=dict(color=color, width=1.6),
            )
        )
    if spy_eq is not None:
        spy = _slice_rebase(spy_eq, window_start, None)
        spy_dd = (spy / spy.cummax() - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=spy_dd.index,
                y=spy_dd.values,
                mode="lines",
                name=spy_label,
                line=dict(color="#ef5350", dash="dash", width=1.5),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#181818",
        margin=dict(l=50, r=180, t=50, b=40),
        height=380,
        yaxis_title="Drawdown (%)",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            bgcolor="rgba(26,26,26,0.8)",
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def cash_mobilization_overlay(
    result: SchemeResult,
    cohort_size: int,
    per_name_cap: float,
    *,
    window_start: pd.Timestamp | None = None,
    div_id: str = "fig-cash-mob",
) -> str:
    """Cash mobilization plot — 4 stacked subplots over time.

    Per ``PORTFOLIO_WEIGHTING_METHODS.md`` §8.1. Visualizes whether cash is
    dynamic (compounding) or static (bug). Constructable from the daily
    snapshot the walker already records.

    A: NAV / cash / deployed (3 lines, $ y-axis) — cash should grow during
       win streaks under cash_fraction.
    B: Utilization (deployed / NAV, %) — healthy band 50-95%.
    C: # active positions (step) with cap-implied max as reference.
    D: Entry health markers — green = clean, yellow = too_small,
       red = clipped (should be empty under cash_fraction).
    """
    if not result.state.daily_snapshot:
        return f"<div id='{div_id}'>(no snapshot data)</div>"
    snap = pd.DataFrame(result.state.daily_snapshot).set_index("date")
    if window_start is not None:
        snap = snap.loc[snap.index >= window_start]
    if snap.empty:
        return f"<div id='{div_id}'>(no snapshot data in window)</div>"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.34, 0.22, 0.22, 0.22],
        vertical_spacing=0.05,
        subplot_titles=(
            "A — NAV decomposition (NAV / cash / deployed)",
            "B — Utilization % (deployed / NAV)",
            "C — Active positions (step)",
            "D — Entry health (green=clean, yellow=too_small, red=clipped)",
        ),
    )

    # Subplot A — NAV / cash / deployed
    fig.add_trace(
        go.Scatter(
            x=snap.index,
            y=snap["netliq"],
            name="NAV",
            line=dict(color="#9ecbff", width=1.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=snap.index,
            y=snap["cash"],
            name="Cash",
            line=dict(color="#26a69a", width=1.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=snap.index,
            y=snap["position_value"],
            name="Deployed",
            line=dict(color="#ffa726", width=1.4),
        ),
        row=1,
        col=1,
    )

    # Subplot B — utilization %
    util = (snap["position_value"] / snap["netliq"].replace(0, np.nan)) * 100
    fig.add_trace(
        go.Scatter(
            x=snap.index,
            y=util,
            name="Util %",
            line=dict(color="#ce93d8", width=1.4),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(
        y0=50,
        y1=95,
        fillcolor="rgba(63,185,80,0.08)",
        line_width=0,
        row=2,
        col=1,
    )

    # Subplot C — # active positions
    max_active = (
        max(1, int(np.floor(cohort_size * per_name_cap)))
        if per_name_cap > 0
        else cohort_size
    )
    fig.add_trace(
        go.Scatter(
            x=snap.index,
            y=snap["n_positions"],
            name="# active",
            line=dict(color="#9ecbff", width=1.4, shape="hv"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=max_active,
        line=dict(color="#888", dash="dash", width=1),
        row=3,
        col=1,
    )

    # Subplot D — entry health markers
    trades = result.state.trades
    failed = result.state.failed_entries
    if window_start is not None:
        trades = [t for t in trades if t.entry_date >= window_start]
        failed = [f for f in failed if f.date >= window_start]

    clean_dates = [t.entry_date for t in trades]
    too_small_dates = [f.date for f in failed if f.reason == "too_small"]
    clipped_dates = [f.date for f in failed if f.reason == "clipped"]
    no_cash_dates = [f.date for f in failed if f.reason == "no_cash"]

    if clean_dates:
        fig.add_trace(
            go.Scatter(
                x=clean_dates,
                y=[1] * len(clean_dates),
                mode="markers",
                name=f"clean ({len(clean_dates)})",
                marker=dict(color="#3fb950", size=5, symbol="circle"),
            ),
            row=4,
            col=1,
        )
    if too_small_dates:
        fig.add_trace(
            go.Scatter(
                x=too_small_dates,
                y=[2] * len(too_small_dates),
                mode="markers",
                name=f"too_small ({len(too_small_dates)})",
                marker=dict(color="#d29922", size=6, symbol="triangle-up"),
            ),
            row=4,
            col=1,
        )
    if clipped_dates:
        fig.add_trace(
            go.Scatter(
                x=clipped_dates,
                y=[3] * len(clipped_dates),
                mode="markers",
                name=f"clipped ({len(clipped_dates)})",
                marker=dict(color="#f85149", size=6, symbol="x"),
            ),
            row=4,
            col=1,
        )
    if no_cash_dates:
        fig.add_trace(
            go.Scatter(
                x=no_cash_dates,
                y=[4] * len(no_cash_dates),
                mode="markers",
                name=f"no_cash ({len(no_cash_dates)})",
                marker=dict(color="#a371f7", size=6, symbol="square"),
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#181818",
        margin=dict(l=50, r=180, t=40, b=40),
        height=720,
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            bgcolor="rgba(26,26,26,0.8)",
            font=dict(size=10),
        ),
        title=f"Cash Mobilization — {result.scheme}",
    )
    fig.update_xaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", range=[0, 105], row=2, col=1)
    fig.update_yaxes(title_text="count", row=3, col=1)
    fig.update_yaxes(
        title_text="status",
        row=4,
        col=1,
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=["clean", "too_small", "clipped", "no_cash"],
        range=[0, 5],
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
    )


# ---------------------------------------------------------------------------
# Verdict + main builder
# ---------------------------------------------------------------------------


def verdict_html(
    summary: pd.DataFrame,
    n_tickers: int,
    min_tickers_gate: int = 10,
) -> str:
    viable = summary.dropna(subset=["oos_sharpe"]).copy()
    if n_tickers < min_tickers_gate or len(viable) == 0:
        return (
            "<div class='verdict red'>No viable scheme: "
            f"n_tickers={n_tickers} &lt; {min_tickers_gate} gate.</div>"
        )
    viable = viable.sort_values("oos_sharpe", ascending=False)
    best = viable.iloc[0]
    cls = (
        "green"
        if best["oos_sharpe"] >= 1.0
        else "amber"
        if best["oos_sharpe"] >= 0.5
        else "red"
    )
    return (
        f"<div class='verdict {cls}'>"
        f"<b>Best under no-rebalance constraints:</b> "
        f"<code>{best['scheme']}</code><br>"
        f"OOS Sharpe = {fmt_num(best['oos_sharpe'], 3)} &nbsp;|&nbsp; "
        f"CAGR = {fmt_pct(best['oos_cagr'])} &nbsp;|&nbsp; "
        f"MaxDD = {fmt_pct(best['oos_max_dd'])} &nbsp;|&nbsp; "
        f"IR vs SPY = {fmt_num(best['ir_vs_spy'], 3)}<br>"
        f"<span style='color:#aaa;font-size:13px'>"
        f"Ranked by OOS Sharpe with min_tickers ≥ {min_tickers_gate} gate. "
        f"Cohort = {n_tickers} tickers."
        f"</span></div>"
    )


def build_html(
    summary: pd.DataFrame,
    results: list[SchemeResult],
    spy_eq: pd.Series | None,
    bh_eq: pd.Series,
    oos_start: pd.Timestamp,
    seed_nav: float,
    n_tickers: int,
    out_path: Path,
    *,
    strategy_label: str,
    sizing_rule: str,
    deployed_scheme: str | None = None,
    vs_old_s5_html: str | None = None,
    bh_label: str | None = None,
    cohort_default_cap: float = DEFAULT_PER_NAME_CAP,
) -> None:
    """Write the canonical 11-section HTML report.

    Strategy-specific bits parameterised:
      strategy_label   — e.g. "OBV-Pivot", appears in <title> + <h1>
      deployed_scheme  — currently deployed scheme name (for the dedicated
                         "comparison vs old S5" block); pass None to skip
      vs_old_s5_html   — opaque HTML block produced by the strategy's own
                         hardcoded reference table (paleologo_verification.md
                         numbers). Pass None to skip.
      bh_label         — legend label for the buy-hold benchmark. Defaults
                         to ``f"{n_tickers}-ticker buy-hold"``.
    """
    summary_sorted = summary.sort_values(
        "oos_sharpe",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    top3 = set(summary_sorted["scheme"].head(3).tolist())

    # Equity-panel scheme set: union of top-10 by OOS Sharpe + top-10 by OOS
    # end-value (2026-05-10 redesign; previously top-10 by Sharpe only).
    # Sharpe-best ≠ end-value-best when MaxDD distributions differ — both
    # rankings carry independent diagnostic value. Up to 20 lines, typically
    # 10-15 after dedup.
    top10_by_sr = summary_sorted["scheme"].head(10).tolist()
    if "oos_end_value" in summary.columns:
        top10_by_end = (
            summary.sort_values("oos_end_value", ascending=False, na_position="last")[
                "scheme"
            ]
            .head(10)
            .tolist()
        )
    else:
        top10_by_end = top10_by_sr
    top_union = list(dict.fromkeys(top10_by_sr + top10_by_end))  # preserve order, dedup

    if bh_label is None:
        bh_label = f"{n_tickers}-ticker buy-hold"

    eq_full_html = equity_overlay(
        results,
        top_union,
        spy_eq,
        bh_eq,
        oos_start,
        f"Equity — full period — top 10 by OOS Sharpe ∪ top 10 by OOS end-value "
        f"+ SPY + {bh_label}",
        bh_label=bh_label,
        div_id="fig-eq-full",
    )
    eq_oos_html = equity_overlay(
        results,
        top_union,
        spy_eq,
        bh_eq,
        oos_start,
        f"Equity — OOS only ({oos_start.date()}+) — top 10 by OOS Sharpe ∪ "
        f"top 10 by OOS end-value, rebased to ${seed_nav:,.0f}",
        bh_label=bh_label,
        window_start=oos_start,
        rebase_to=seed_nav,
        div_id="fig-eq-oos",
    )
    dd_full_html = drawdown_overlay(
        results,
        top10_by_sr,
        spy_eq,
        "Drawdown — full period — top 10 by OOS Sharpe vs SPY",
        div_id="fig-dd-full",
    )
    dd_oos_html = drawdown_overlay(
        results,
        top10_by_sr,
        spy_eq,
        f"Drawdown — OOS only ({oos_start.date()}+) vs SPY",
        window_start=oos_start,
        div_id="fig-dd-oos",
    )

    # Cash mobilization plots — top OOS-Sharpe scheme + deployed scheme
    # (per PORTFOLIO_WEIGHTING_METHODS.md §8.1). Constructable from the
    # daily snapshot the walker already records — no extra computation.
    by_name = {r.scheme: r for r in results}
    cash_mob_targets: list[tuple[str, str]] = []
    top_scheme_name = top10_by_sr[0] if top10_by_sr else None
    if top_scheme_name and top_scheme_name in by_name:
        cash_mob_targets.append((top_scheme_name, "top OOS-Sharpe"))
    if (
        deployed_scheme is not None
        and deployed_scheme in by_name
        and deployed_scheme != top_scheme_name
    ):
        cash_mob_targets.append((deployed_scheme, "deployed"))
    cash_mob_blocks = []
    for scheme_name, label in cash_mob_targets:
        r = by_name[scheme_name]
        cap = scheme_per_name_cap(scheme_name, cohort_default_cap)
        cash_mob_blocks.append(
            f"<h3 style='color:#9ecbff;font-size:14px;margin-top:18px'>"
            f"{scheme_name} ({label})</h3>"
            + cash_mobilization_overlay(
                r,
                cohort_size=n_tickers,
                per_name_cap=cap,
                div_id=f"fig-cash-mob-{scheme_name}",
            )
        )
    cash_mob_html = (
        "\n".join(cash_mob_blocks)
        if cash_mob_blocks
        else ("<p>(no scheme available for cash mobilization plot)</p>")
    )

    spy_row = spy_summary_row(
        spy_eq, oos_start, list(summary_sorted.columns), seed_nav=seed_nav
    )
    table_html = summary_table_html(summary_sorted, top3, spy_row=spy_row)
    verdict_block = verdict_html(summary_sorted, n_tickers)

    head = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>{strategy_label} — No-Rebalance S5 Replay ({sizing_rule})</title>
<style>
body {{ background:#111; color:#ddd; font-family:system-ui,-apple-system,sans-serif;
        margin:0; padding:20px; }}
h1 {{ color:#fff; margin-bottom:4px; border-bottom:1px solid #333; padding-bottom:10px; }}
h2 {{ color:#fff; margin:28px 0 10px 0; font-size:16px; }}
.sub {{ color:#888; margin-bottom:18px; font-size:13px; }}
.summary-table {{ border-collapse:collapse; font-size:12px; width:100%;
                    margin-bottom:12px; }}
.summary-table th {{ background:#222; color:#fff; padding:8px; text-align:left;
                      border-bottom:1px solid #444; cursor:pointer;
                      user-select:none; font-variant-numeric:tabular-nums; }}
.summary-table th:hover {{ background:#2a2a2a; }}
.summary-table td {{ padding:5px 8px; border-bottom:1px solid #222;
                      font-variant-numeric:tabular-nums; text-align:right; }}
.summary-table td:first-child {{ text-align:left; font-weight:600;
                                   color:#9ecbff; }}
.summary-table tr.row-top td {{ background:#1a2a1a; color:#26a69a; }}
.summary-table th.cell-spy {{ color:#f85149; }}
.summary-table td.cell-spy {{ color:#f85149; }}
.summary-table tr.row-top td.cell-spy {{ color:#ff6e6e; }}
.summary-table tr.row-spy td {{
    background:#2a0d0d; color:#f85149; font-weight:600;
    border-bottom:2px solid #f85149;
}}
.summary-table tr.row-spy td:first-child {{ color:#ff6e6e; }}
.summary-table tr.row-spy td.cell-spy {{ color:#ff9999; }}
.plot-block {{ background:#111; margin-bottom:14px; }}
.panel {{ background:#161b22; border:1px solid #30363d; border-radius:6px;
          padding:14px; margin:14px 0; }}
.verdict {{ padding:14px; border-radius:6px; font-size:15px; }}
.verdict.green {{ background:#0d3a1e; border-left:5px solid #3fb950; }}
.verdict.amber {{ background:#3a2a0d; border-left:5px solid #d29922; }}
.verdict.red {{ background:#3a0d0d; border-left:5px solid #f85149; }}
code {{ background:#222; padding:2px 6px; border-radius:3px; }}
a {{ color:#58a6ff; }}
</style></head><body>
<h1>{strategy_label} — No-Rebalance S5 Replay (24 schemes) — {sizing_rule}</h1>
<div class='sub'>
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
  Seed NAV <b>${seed_nav:,.0f}</b> ·
  Cohort {n_tickers} tickers ·
  OOS start {oos_start.date()} ·
  Sizing rule <code>{sizing_rule}</code>
</div>

<div class='panel'>
  <b>Why this exists.</b> The S5 framework in
  <code>build_portfolio.py</code> aggregates per-ticker returns under daily
  rebalance, an assumption live execution cannot satisfy without infinite
  cash. This replay redoes S5 across all 24 schemes with realistic
  entry-time sizing and hold-to-exit semantics. See
  <code>PORTFOLIO_WEIGHTING_METHODS.md</code> for the canonical 4-layer
  model and methodology details.
  <br><br>
  <b>Sizing rule:</b> <code>netliq_clip</code> targets w×NetLiq and clips
  at cash×buffer (legacy Option B); <code>cash_fraction</code> targets
  w×cash×buffer with cash snapshot before the entry-sort loop, guaranteeing
  zero clipped/no-cash failures by construction (Σw_entry ≤ 1.0 over the
  active set).
</div>
"""

    deployed_label = deployed_scheme if deployed_scheme is not None else "(none)"
    vs_old_block = (
        vs_old_s5_html
        if vs_old_s5_html is not None
        else "<p>(no reference comparison provided)</p>"
    )

    body = f"""
{verdict_block}

<h2>Scheme comparison — sortable (OOS + full-period + vs-SPY + execution)</h2>
<div class='sub'>Click a column header to sort. Top 3 by OOS Sharpe
highlighted (green). The <span style='color:#f85149;font-weight:600'>SPY
benchmark row (red)</span> is pinned at the top regardless of sort —
direct visual comparison against every scheme. OOS metrics use per-pair
OOS dates (post {oos_start.date()}); full-period spans the entire walk.
SPY-comparison columns (ir_vs_spy, alpha_ann, beta, ...) shown in red text.
<code>full_*</code> columns are new in the 2026-04-30 audit revision.</div>
{table_html}

<h2>Equity curves — full period — top 10 by OOS Sharpe ∪ top 10 by OOS end-value</h2>
<div class='sub'>Sharpe-best ≠ end-value-best when MaxDD distributions differ; both rankings shown together.</div>
<div class='plot-block'>{eq_full_html}</div>

<h2>Equity curves — OOS only ({oos_start.date()}+, rebased) — top 10 by OOS Sharpe ∪ top 10 by OOS end-value</h2>
<div class='sub'>Portfolio + SPY + buy-hold all rebased to seed NAV at OOS start so the post-OOS alpha is directly comparable.</div>
<div class='plot-block'>{eq_oos_html}</div>

<h2>Drawdown — full period</h2>
<div class='plot-block'>{dd_full_html}</div>

<h2>Drawdown — OOS only ({oos_start.date()}+)</h2>
<div class='sub'>Cummax computed within the OOS window only — does not
carry over pre-OOS peaks.</div>
<div class='plot-block'>{dd_oos_html}</div>

<h2>Cash Mobilization — is cash dynamic or static?</h2>
<div class='sub'>
  Per <code>PORTFOLIO_WEIGHTING_METHODS.md</code> §8.1. Four stacked
  subplots over the full date range:
  <b>A</b> NAV / cash / deployed (compounding visible if cash grows during
  win streaks);
  <b>B</b> Utilization % (deployed/NAV — healthy band 50-95% shaded green);
  <b>C</b> Active position count (dashed line = cohort × cap, theoretical max);
  <b>D</b> Entry health markers — green=clean (cash_fraction sized OK),
  yellow=too_small (rounded to &lt;1 share), red=clipped (only under
  netliq_clip), purple=no_cash. Under <code>cash_fraction</code> there
  should be zero red dots by construction.
</div>
{cash_mob_html}

<h2>Comparison vs old S5 (deployed scheme: {deployed_label})</h2>
{vs_old_block}

<p style='color:#666;font-size:11px;margin-top:30px'>
Generated by no_rebalance_replay.py
</p>
</body></html>
"""
    out_path.write_text(head + body)
