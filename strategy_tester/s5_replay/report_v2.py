"""report_v2 — tabbed mega-report on quant-ui-kit, with per-scheme picker.

Single self-contained HTML with 11 tabs covering the full analytics surface
(quantstats / pyfolio / qf-lib ports). A `<select>` at the top lets you
switch among the top-N schemes by OOS Sharpe and the entire dashboard
(KPI cards, metric tiles, every Plotly chart in tabs 2-9, DD table, stress
table, Trades tab, cash mobilization) re-renders for the chosen scheme.

Tabs: Overview / Quantstats / Risk / Rolling / Distribution / Benchmark /
      Bootstrap / Stress / Trades / Execution / Cohort.

Implementation:
  - Plotly charts → embedded as JSON in `SCHEME_FIGS = {div_id: {scheme: fig}}`,
    swapped via `Plotly.react()` on dropdown change.
  - HTML fragments (KPI cards, metric tiles, DD periods table, stress table,
    trades KPI / summary / full-log tables, cash mobilization) → all N
    rendered into the same DOM with `<div data-scheme="X">` wrappers; only
    the active one is visible.
  - Tab-1 multi-scheme overlays + Cohort tab stay scheme-independent.

CSS+JS from `reports/quant-ui-kit/` are inlined at build time; Plotly is
loaded from CDN.

────────────────────────────────────────────────────────────────────────────
## How to use from a strategy pipeline

After your strategy's S5 run completes, add this block to its pipeline.py
(typically right after the existing v1 `build_html()` call so both reports
ship side-by-side during the transition):

```python
from pathlib import Path
from strategy_tester.s5_replay.runner import run_all_schemes
from strategy_tester.s5_replay.metrics import summary_rows
from strategy_tester.s5_replay.report_v2 import build_mega_html
from strategy_tester.s5_replay.io_state import save_scheme_state_bundle

# ── 1. Run all schemes (existing call — unchanged) ──────────────────────
results = run_all_schemes(
    schemes_to_run=schemes, cache=cache, base_maps=base_maps,
    rolling_sr=rolling_sr, seed_nav=SEED_NAV, buffer=0.95,
    spy_eq=spy_eq, oos_start=oos_start, workers=8,
)
summary = summary_rows(results, spy_eq, oos_start)

# ── 2. Build the mega-report (single self-contained HTML) ──────────────
build_mega_html(
    summary=summary,
    results=results,
    spy_eq=spy_eq,
    bh_eq=bh_eq,
    oos_start=oos_start,
    seed_nav=SEED_NAV,
    n_tickers=len(cohort),
    out_path=results_dir / "per_asset_s5_v2_combined_latest.html",
    strategy_label="<strategy_name>",       # appears in <title> + <h1>
    sizing_rule="cash_fraction",
    deployed_scheme=deployed_scheme,        # currently-deployed scheme name
    bh_label=f"{len(cohort)}-ticker buy-hold",
    top_n_picker=5,                         # default 5 → ~50 MB; max 10
)

# ── 3. (Optional) Persist state for offline re-rendering ────────────────
# Saves <scheme>_{snapshots,trades,failed_entries}.parquet so you can
# rebuild the report later without re-running the backtest.
state_dir = results_dir / "scheme_state"
for r in results[:5]:  # top-5 by OOS Sharpe = picker set
    save_scheme_state_bundle(r.state, state_dir, r.scheme)
```

The walker now emits `mae_pct / mfe_pct / hold_days / commission_dollars`
on every closed `TradeLog`, so the Trades tab populates automatically — no
strategy-side changes needed beyond the call above.

If your strategy's S5 emits an equity-only parquet (no live `SchemeResult`
in memory), you can fabricate a minimal one for the report — see
`/tmp/render_canary.py` for the pattern, or load persisted state via
`io_state.load_*` helpers.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

from strategy_tester.s5_replay.charts.annual import (
    annual_returns_bar_figure,
    log_equity_figure,
    snapshot_3panel_figure,
)
from strategy_tester.s5_replay.charts.benchmark import (
    cumulative_excess_figure,
    regression_scatter_figure,
    rolling_correlation_figure,
    spy_decile_returns_figure,
    up_down_capture_figure,
)
from strategy_tester.s5_replay.charts.bootstrap import (
    forecast_cone_figure,
    monte_carlo_paths_figure,
    perf_stats_box_figure,
)
from strategy_tester.s5_replay.charts.distribution import (
    kde_vs_normal_figure,
    returns_boxplot_figure,
    returns_histogram_figure,
)
from strategy_tester.s5_replay.charts.drawdown import (
    drawdown_table_html,
    top_drawdowns_highlight_figure,
    underwater_figure,
)
from strategy_tester.s5_replay.charts.heatmaps import (
    monthly_heatmap_figure,
    scheme_correlation_heatmap_html,
)
from strategy_tester.s5_replay.charts.rolling import (
    rolling_beta_figure,
    rolling_sharpe_figure,
    rolling_sortino_figure,
    rolling_volatility_figure,
)
from strategy_tester.s5_replay.charts.stress import (
    stress_grid_figure,
    stress_table_html,
)
from strategy_tester.s5_replay.charts.trades import (
    exit_reason_breakdown_figure,
    hold_time_distribution_figure,
    mae_mfe_scatter_figure,
    per_symbol_pnl_bar_figure,
    pnl_distribution_figure,
    round_trip_lifetimes_figure,
    trades_full_table_html,
    trades_kpi_strip_html,
    trades_summary_table_html,
    trades_to_dataframe,
)
from strategy_tester.s5_replay.extra_metrics import (
    extended_metrics,
    extended_vs_benchmark,
)
from strategy_tester.s5_replay.metrics import (
    equity_metrics,
    spy_summary_row,
    vs_spy,
)
from strategy_tester.s5_replay.oracles import scheme_per_name_cap
from strategy_tester.s5_replay.report import (
    cash_mobilization_overlay,
    drawdown_overlay,
    equity_overlay,
    verdict_html,
)
from strategy_tester.s5_replay.runner import SchemeResult
from strategy_tester.s5_replay.walker import DEFAULT_PER_NAME_CAP

_KIT = (
    Path(__file__).resolve().parents[2] / ".." / "reports" / "quant-ui-kit"
).resolve()

# How many top schemes the picker exposes. `None` = ALL schemes (post-2026-05-18
# ref-doc c73 "S5 should sweep all schemes and surface everything"). Default
# remains 5 (~50 MB HTML) for backward-compat with existing callers; pass
# `top_n_picker=None` (or `len(results)`) to render all schemes in the picker.
# File size scales linearly with picker size (~10 MB per scheme).
DEFAULT_TOP_N_PICKER = 5

# Compact view columns for the scheme comparison table (Tab 1). Concentration
# + cap-binding diagnostics added 2026-05-18 (ref-doc Q5/Q17/Q21). The
# `sizing_rule` column appears only when caller passes a summary that contains
# it (combined multi-rule HTMLs); single-rule summaries skip it gracefully via
# the "if c in summary.columns" filter.
_CORE_COLS: list[str] = [
    "scheme", "n_aliases", "sizing_rule",
    "oos_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar",
    "oos_sortino", "oos_vol_ann", "oos_end_value",
    "full_sharpe",
    "ir_vs_spy", "alpha_ann", "beta", "excess_cagr",
    "n_trades",
    "mean_enb", "max_hhi", "cap_bound_rate",
    "p95_entry_pct_cash", "max_entry_pct_cash", "n_too_small",
]
_GROUP_STARTS = {
    "oos_sharpe": "OOS",
    "full_sharpe": "Full",
    "ir_vs_spy": "vs SPY",
    "n_trades": "Exec",
    "mean_enb": "Concentration",
    "p95_entry_pct_cash": "Cap diag",
}


def _strip_rule_tag(scheme: str) -> str:
    """Strip a leading `{rule_tag}__` prefix from a scheme name. Tags are
    short codes (`cf`, `cfc`, `seq`, `plg`) used by the combined-multi-rule
    HTMLs to keep picker keys unique. Used before passing the scheme to
    ``scheme_per_name_cap()`` which only knows the unprefixed names.
    """
    if "__" not in scheme:
        return scheme
    prefix, _, tail = scheme.partition("__")
    if prefix in {"cf", "cfc", "seq", "plg"}:
        return tail
    return scheme


def _curve_hash(equity: pd.Series, oos_start: pd.Timestamp) -> str | None:
    """SHA-1 of the rebased OOS equity curve, rounded to 6 decimals of
    relative precision. Identical curves → identical hash; FP-noise differences
    are absorbed by the rounding. Returns None if the curve is empty / zero-start
    (un-hashable, treated as unique by the caller).
    """
    eq_oos = equity.loc[equity.index >= oos_start]
    if len(eq_oos) == 0:
        return None
    arr = eq_oos.to_numpy(dtype=np.float64)
    if arr.size == 0 or arr[0] == 0.0:
        return None
    norm = arr / arr[0]
    rounded = np.around(norm, decimals=6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def _tiebreak_key(scheme: str, deployed: str | None) -> tuple:
    """Ordering for picking the representative within a dedup group. Lower wins.

    Priority (per Daniel 2026-05-18):
      1. deployed scheme always wins.
      2. simpler name: prefer schemes WITHOUT a `_capXX` suffix; within
         cap-suffix schemes, prefer shorter name.
      3. lowest cap value: `cap05 < cap10 < ... < cap50`.
      4. alphabetical.
    """
    stripped = _strip_rule_tag(scheme)
    has_cap = "_cap" in stripped
    cap_val = 100
    if has_cap:
        tail = stripped[-2:]
        if tail.isdigit():
            cap_val = int(tail)
    return (
        scheme != deployed,   # False (=0) wins → deployed first
        int(has_cap),         # 0 (no cap) wins → simpler concept
        len(stripped),        # shorter wins
        cap_val,              # lower cap wins
        scheme,               # alpha
    )


def _filter_and_dedup(
    summary: pd.DataFrame,
    results: list,
    oos_start: pd.Timestamp,
    deployed_scheme: str | None,
    exclude_cf_family: bool,
    dedup_duplicates: bool,
) -> tuple[pd.DataFrame, list, dict[str, list[str]]]:
    """Return (summary_filtered, results_filtered, aliases).

    ``aliases`` maps each surviving scheme → list of dropped scheme names
    that collapsed into it (used for the ≡ column in the comparison table).

    Two filters, applied in order:
      1. ``exclude_cf_family``: drop schemes prefixed `cf__` (uncapped
         sizing rule — comparison baseline, not deployable; max_entry
         caps at the 95% buffer ceiling and collides with every other
         family on single-signal bars).
      2. ``dedup_duplicates``: hash each surviving scheme's rebased OOS
         equity curve; collapse hash-equal groups to one representative
         picked by ``_tiebreak_key``.
    """
    aliases: dict[str, list[str]] = {}
    surviving = list(results)

    if exclude_cf_family:
        surviving = [r for r in surviving if not r.scheme.startswith("cf__")]

    if dedup_duplicates:
        groups: dict[str, list] = {}
        unhashable: list = []
        for r in surviving:
            h = _curve_hash(r.equity, oos_start)
            if h is None:
                unhashable.append(r)
                continue
            groups.setdefault(h, []).append(r)
        kept: list = []
        for members in groups.values():
            if len(members) == 1:
                kept.append(members[0])
                continue
            members_sorted = sorted(
                members,
                key=lambda r: _tiebreak_key(r.scheme, deployed_scheme),
            )
            rep = members_sorted[0]
            aliases[rep.scheme] = [m.scheme for m in members_sorted[1:]]
            kept.append(rep)
        kept.extend(unhashable)
        surviving = kept

    surviving_names = [r.scheme for r in surviving]
    mask = summary["scheme"].isin(surviving_names)
    summary_filtered = summary.loc[mask].copy()
    summary_filtered["n_aliases"] = [
        len(aliases.get(s, [])) for s in summary_filtered["scheme"]
    ]
    summary_filtered["aliases_list"] = [
        ", ".join(aliases.get(s, [])) for s in summary_filtered["scheme"]
    ]
    return summary_filtered, surviving, aliases


def _inline_assets() -> tuple[str, str]:
    css = (_KIT / "quant-ui.css").read_text()
    js = (_KIT / "quant-ui.js").read_text()
    return css, js


# ---------------------------------------------------------------------------
# Compact comparison table (Tab 1) — sparkline + group headers, unchanged.
# ---------------------------------------------------------------------------


def _sparkline_svg(
    eq: pd.Series, width: int = 110, height: int = 26,
) -> str:
    if eq is None or len(eq) < 2:
        return ""
    vals = eq.values.astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return ""
    n = len(vals)
    points = []
    for i, v in enumerate(vals):
        x = i / (n - 1) * (width - 2) + 1
        y = height - 1 - (v - lo) / (hi - lo) * (height - 2)
        points.append(f"{x:.1f},{y:.1f}")
    color = "#3fb950" if vals[-1] >= vals[0] else "#ef5350"
    fill = (
        "rgba(63,185,80,0.18)" if vals[-1] >= vals[0]
        else "rgba(239,83,80,0.18)"
    )
    area = points + [f"{width - 1:.1f},{height - 1:.1f}", f"1,{height - 1:.1f}"]
    return (
        f'<svg width="{width}" height="{height}" '
        f'style="display:block;vertical-align:middle">'
        f'<polyline fill="{fill}" stroke="none" '
        f'points="{" ".join(area)}"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.4" '
        f'points="{" ".join(points)}"/>'
        f'</svg>'
    )


_COMPACT_PCT_DEC = {
    "oos_cagr", "oos_max_dd", "oos_vol_ann", "alpha_ann", "excess_cagr",
    "cap_bound_rate",
}
_COMPACT_NUM3 = {
    "oos_sharpe", "oos_calmar", "oos_sortino", "full_sharpe",
    "ir_vs_spy", "beta",
    "mean_enb", "max_hhi",
}
_COMPACT_MONEY = {"oos_end_value"}
_COMPACT_INT = {"n_trades", "n_too_small", "n_aliases"}
_COMPACT_NUM2 = {"p95_entry_pct_cash", "max_entry_pct_cash"}
_COMPACT_HEADERS = {
    "scheme": "Scheme", "n_aliases": "≡", "sizing_rule": "Sizing rule",
    "oos_sharpe": "Sharpe", "oos_cagr": "CAGR", "oos_max_dd": "MaxDD",
    "oos_calmar": "Calmar", "oos_sortino": "Sortino", "oos_vol_ann": "Vol",
    "oos_end_value": "End $",
    "full_sharpe": "Sharpe",
    "ir_vs_spy": "IR", "alpha_ann": "Alpha", "beta": "Beta",
    "excess_cagr": "ΔCAGR",
    "n_trades": "Trades",
    "mean_enb": "ENB", "max_hhi": "max HHI", "cap_bound_rate": "Cap %",
    "p95_entry_pct_cash": "p95 ent%", "max_entry_pct_cash": "max ent%",
    "n_too_small": "TooSm",
}


def _fmt_compact_cell(col: str, v) -> tuple[str, str, str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—", "-1e9", "muted"
    if col == "scheme" or col == "sizing_rule":
        return str(v), str(v), ""
    if not isinstance(v, (int, float)):
        return str(v), str(v), ""
    fv = float(v)
    if col in _COMPACT_PCT_DEC:
        tone = "pos" if fv > 0 else "neg" if fv < 0 else "muted"
        if col == "oos_max_dd":
            tone = "neg"
        return f"{fv * 100:+.2f}%", str(fv), tone
    if col in _COMPACT_NUM3:
        tone = "pos" if fv > 0 else "neg" if fv < 0 else "muted"
        return f"{fv:.3f}", str(fv), tone
    if col in _COMPACT_MONEY:
        return f"${fv:,.0f}", str(fv), ""
    if col in _COMPACT_INT:
        return f"{int(fv):,}", str(fv), ""
    if col in _COMPACT_NUM2:
        return f"{fv:.2f}", str(fv), ""
    return f"{fv:.4f}", str(fv), ""


def _compact_table_html(
    summary: pd.DataFrame,
    top3: set[str],
    spy_row: dict | None,
    equity_by_scheme: dict[str, pd.Series],
    spy_eq: pd.Series | None,
    oos_start: pd.Timestamp,
    ir_floor: float = 0.0,
) -> str:
    # Drop rows with IR vs SPY below ``ir_floor`` (2026-05-18; ir_floor param
    # added 2026-05-22). Schemes that underperform SPY on the information-
    # ratio dimension add row noise without decision value — they fail
    # Carver / Grinold-Kahn's minimum gate. SPY itself (IR=0 vs itself) is
    # rendered separately above the body, so this filter doesn't drop the
    # benchmark row. Set ``ir_floor`` to a very negative number (e.g. -999)
    # to disable the filter — useful when no scheme beats SPY and you still
    # want to see all rows for exploration.
    n_before_filter = len(summary)
    if "ir_vs_spy" in summary.columns:
        summary = summary[summary["ir_vs_spy"].fillna(-1.0) >= ir_floor].copy()
    n_after_filter = len(summary)
    cols = [c for c in _CORE_COLS if c in summary.columns]
    # Drop ≡ alias-count column when nothing was deduped (avoids a column
    # of zeros / blanks that adds noise to every row).
    if "n_aliases" in cols:
        if "n_aliases" in summary.columns and len(summary) > 0:
            max_aliases = int(summary["n_aliases"].fillna(0).max())
        else:
            max_aliases = 0
        if max_aliases == 0:
            cols.remove("n_aliases")
    head_cells = []
    for c in cols:
        cls = "group-start" if c in _GROUP_STARTS else ""
        label = _COMPACT_HEADERS.get(c, c.replace("_", " "))
        data_type = (
            "money" if c in _COMPACT_MONEY
            else "pct" if c in _COMPACT_PCT_DEC
            else "num" if c in (_COMPACT_NUM3 | _COMPACT_INT)
            else "text"
        )
        group_label = (
            f"<div class='group-label'>{_GROUP_STARTS[c]}</div>"
            if c in _GROUP_STARTS else ""
        )
        head_cells.append(
            f"<th data-col='{c}' data-type='{data_type}' "
            f"data-align='right' class='{cls}'>"
            f"{group_label}<div class='col-label'>{label}</div></th>"
        )
    head_cells.append(
        "<th data-col='spark' data-type='num' data-align='left' "
        "class='group-start'><div class='group-label'>OOS</div>"
        "<div class='col-label'>Curve</div></th>"
    )
    head = "<thead><tr>" + "".join(head_cells) + "</tr></thead>"

    def _build_row(scheme: str, src: dict, is_spy: bool) -> str:
        cells = []
        for c in cols:
            v = src.get(c)
            disp, sort_val, tone = _fmt_compact_cell(c, v)
            tone_attr = f"data-tone='{tone}'" if tone else ""
            cls = "group-start" if c in _GROUP_STARTS else ""
            # n_aliases cells: render '≡N' (or '—' when N=0) with a
            # tooltip listing the absorbed scheme names so users can audit
            # what the dedup collapsed.
            title_attr = ""
            if c == "n_aliases":
                try:
                    n = int(float(sort_val))
                except (TypeError, ValueError):
                    n = 0
                if n > 0:
                    disp = f"≡{n}"
                    alist = src.get("aliases_list") or ""
                    title_attr = f"title='aliases: {alist}'"
                    tone_attr = "data-tone='accent'"
                else:
                    disp = "—"
                    tone_attr = "data-tone='muted'"
            cells.append(
                f"<td data-val='{sort_val}' data-col='{c}' "
                f"class='{cls}' {tone_attr} {title_attr}>{disp}</td>"
            )
        if is_spy and spy_eq is not None:
            spark_eq = spy_eq.loc[spy_eq.index >= oos_start]
        else:
            eq = equity_by_scheme.get(scheme)
            spark_eq = (
                eq.loc[eq.index >= oos_start] if eq is not None
                else pd.Series(dtype=float)
            )
        spark = _sparkline_svg(spark_eq, width=110, height=26)
        spark_sort_val = (
            float(spark_eq.iloc[-1]) if not spark_eq.empty else 0.0
        )
        cells.append(
            f"<td data-val='{spark_sort_val}' class='group-start spark-cell'>"
            f"{spark}</td>"
        )
        cls = "row-top" if scheme in top3 else ""
        if is_spy:
            # `row-spy` styles the row; `pinned-top` is the class
            # quant-ui-kit's sortable JS uses to keep rows at the top
            # during column sort. Both classes needed.
            cls = "row-spy pinned-top"
        pinned = "data-pinned='1'" if is_spy else ""
        return f"<tr class='{cls}' {pinned}>" + "".join(cells) + "</tr>"

    body_rows = []
    if spy_row is not None:
        body_rows.append(_build_row("SPY", spy_row, is_spy=True))
    for _, r in summary.iterrows():
        body_rows.append(_build_row(r["scheme"], dict(r), is_spy=False))
    body = "<tbody>" + "\n".join(body_rows) + "</tbody>"

    style = """
<style>
  /* Sticky thead + sticky SPY row (2026-05-18). Outer wrap allows
     both horizontal AND vertical scroll; the header + SPY benchmark
     stay pinned at the top while the body scrolls. */
  .compact-tbl-wrap {
    overflow: auto; max-height: 72vh;
    position: relative;
  }
  .compact-tbl { min-width: max-content; border-collapse: separate;
    border-spacing: 0; }
  .compact-tbl th, .compact-tbl td {
    padding: 9px 14px !important; white-space: nowrap;
  }
  .compact-tbl thead th {
    font-size: 12px; letter-spacing: 0.02em; vertical-align: bottom;
    position: sticky; top: 0; z-index: 10;
    background: var(--bg-0, #0d1117);
    box-shadow: inset 0 -1px 0 var(--border, #30363d);
  }
  .compact-tbl th .group-label {
    font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.10em;
    color: var(--muted, #8b949e); text-align: center; padding-bottom: 4px;
    font-weight: 600; border-bottom: 1px solid var(--border, #30363d);
    margin-bottom: 4px;
  }
  .compact-tbl th .col-label { display: block; text-align: right; }
  .compact-tbl td { font-size: 12.5px; line-height: 1.35; }
  .compact-tbl .group-start {
    border-left: 1px solid var(--border-strong, #30363d) !important;
  }
  .compact-tbl .spark-cell { padding: 4px 8px !important; width: 120px; }
  .compact-tbl tr.row-top td { background: rgba(63,185,80,0.06); }
  /* SPY row pinned just below the sticky header. Each cell carries its
     own position:sticky because position:sticky on <tr> doesn't work
     reliably across browsers. The 56px top offset matches the header
     height (~2 lines: group-label + col-label). */
  .compact-tbl tr.row-spy td {
    background: rgba(239,83,80,0.18);
    border-top: 1px solid rgba(239,83,80,0.35);
    border-bottom: 1px solid rgba(239,83,80,0.35);
    position: sticky; top: 56px; z-index: 9;
    font-weight: 600;
  }
</style>
"""
    note = (
        f"<span class='section-note' style='color:var(--text-muted); "
        f"font-size:0.82rem; margin-left:0.6rem;'>"
        f"filter: <code>IR vs SPY &ge; 0</code> &mdash; "
        f"{n_after_filter} of {n_before_filter} rows shown "
        f"(SPY pinned)</span>"
    )
    return (
        f"{style}{note}<div class='compact-tbl-wrap'>"
        f"<table class='q-table sortable compact-tbl' id='cmp-tbl'>"
        f"{head}{body}</table></div>"
    )


# ---------------------------------------------------------------------------
# Per-scheme HTML fragment builders (KPI cards + metric tiles)
# ---------------------------------------------------------------------------


def _fmt_pct(v: float, sign: bool = True) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:+.2f}%" if sign else f"{v * 100:.2f}%"


def _fmt_num(v: float, dp: int = 3) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{dp}f}"


def _tone_for_metric(name: str, v: float) -> str:
    if v is None or pd.isna(v):
        return "muted"
    if name in {"max_dd", "var_95", "var_99", "cvar_95", "cvar_99"}:
        return "neg" if v < 0 else "pos"
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return "muted"


def _kpi_card(label: str, value: str, tone: str, sub: str = "") -> str:
    return (
        f"<div class='kpi-card' data-tone='{tone}'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value' data-tone='{tone}'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>"
    )


def _metric_card(label: str, value: str, tone: str = "") -> str:
    tone_attr = f"data-tone='{tone}'" if tone else ""
    return (
        f"<div class='metric-card'>"
        f"<span class='metric-label'>{label}</span>"
        f"<span class='metric-value' {tone_attr}>{value}</span></div>"
    )


def _kpi_block(
    scheme_name: str, base_oos: dict, ext_oos: dict, vs_oos: dict,
    n_total_schemes: int, seed_nav: float, n_tickers: int,
) -> str:
    return (
        "<div class='kpi-grid'>"
        + _kpi_card(
            "Selected scheme", scheme_name, "accent",
            f"out of {n_total_schemes} by OOS Sharpe",
        )
        + _kpi_card(
            "OOS Sharpe", _fmt_num(base_oos.get("sharpe", float("nan")), 3),
            _tone_for_metric("sharpe", base_oos.get("sharpe", 0)),
            f"PSR = {_fmt_num(ext_oos.get('probabilistic_sharpe', float('nan')), 3)}",
        )
        + _kpi_card(
            "OOS CAGR", _fmt_pct(base_oos.get("cagr", float("nan"))),
            "pos" if base_oos.get("cagr", 0) > 0 else "neg",
            f"vs SPY excess: {_fmt_pct(vs_oos.get('excess_cagr', float('nan')))}",
        )
        + _kpi_card(
            "OOS MaxDD", _fmt_pct(base_oos.get("max_dd", float("nan"))), "neg",
            f"Calmar {_fmt_num(base_oos.get('calmar', float('nan')), 2)}"
            f" · Ulcer {_fmt_num(base_oos.get('ulcer', float('nan')), 2)}",
        )
        + _kpi_card(
            "IR vs SPY", _fmt_num(vs_oos.get("ir_vs_spy", float("nan")), 3),
            "pos" if vs_oos.get("ir_vs_spy", 0) > 0 else "neg",
            f"α(ann)={_fmt_pct(vs_oos.get('alpha_ann', float('nan')))}"
            f" · β={_fmt_num(vs_oos.get('beta', float('nan')), 2)}",
        )
        + _kpi_card(
            "Cohort", f"{n_tickers:,}", "muted",
            f"seed NAV ${seed_nav:,.0f}",
        )
        + "</div>"
    )


def _metrics_block(
    base: dict, extra: dict, vs_b: dict, extra_b: dict,
) -> str:
    groups: list[tuple[str, list[tuple[str, str, str]]]] = []

    def pct(k, v, sign=True):
        return (k, _fmt_pct(v, sign=sign), _tone_for_metric(k, v))

    def num3(k, v, dp=3):
        return (k, _fmt_num(v, dp=dp), _tone_for_metric(k, v))

    def int_(k, v):
        return (k, f"{int(v) if pd.notna(v) else 0:,}", "")

    groups.append(("Returns", [
        pct("CAGR", base.get("cagr", float("nan"))),
        pct("Total Ret", extra.get("total_return", float("nan"))),
        pct("Best day", extra.get("best_day", float("nan"))),
        pct("Worst day", extra.get("worst_day", float("nan"))),
        pct("Avg win", extra.get("avg_win", float("nan"))),
        pct("Avg loss", extra.get("avg_loss", float("nan"))),
        pct("Avg return", extra.get("avg_return", float("nan"))),
        pct("Geo mean", extra.get("geometric_mean", float("nan"))),
        pct("RAR", extra.get("rar", float("nan"))),
    ]))
    groups.append(("Ratios", [
        num3("Sharpe", base.get("sharpe", float("nan"))),
        num3("Smart Sharpe", extra.get("smart_sharpe", float("nan"))),
        num3("Sortino", base.get("sortino", float("nan"))),
        num3("Smart Sortino", extra.get("smart_sortino", float("nan"))),
        num3("Adj Sortino", extra.get("adjusted_sortino", float("nan"))),
        num3("Calmar", base.get("calmar", float("nan"))),
        num3("Omega", extra.get("omega", float("nan"))),
        num3("Martin", base.get("martin", float("nan"))),
        num3("UPI", extra.get("upi", float("nan"))),
        num3("Serenity", extra.get("serenity_index", float("nan"))),
        num3("Gain-to-Pain", extra.get("gain_to_pain", float("nan"))),
        num3("Risk-Ret Ratio", extra.get("risk_return_ratio", float("nan"))),
        num3("Payoff", extra.get("payoff_ratio", float("nan"))),
        num3("Win/Loss", extra.get("win_loss_ratio", float("nan"))),
        num3("CPC Index", extra.get("cpc_index", float("nan"))),
        num3("Common-Sense", extra.get("common_sense_ratio", float("nan"))),
        num3("Recovery", extra.get("recovery_factor", float("nan"))),
    ]))
    groups.append(("Risk / Tail", [
        pct("Max DD", base.get("max_dd", float("nan"))),
        pct("Ann Vol", base.get("vol_ann", float("nan")), sign=False),
        pct("VaR 95%", extra.get("var_95", float("nan"))),
        pct("VaR 99%", extra.get("var_99", float("nan"))),
        pct("CVaR 95%", extra.get("cvar_95", float("nan"))),
        pct("CVaR 99%", extra.get("cvar_99", float("nan"))),
        num3("Tail ratio", extra.get("tail_ratio", float("nan")), dp=2),
        num3("Outlier W", extra.get("outlier_win_ratio", float("nan")), dp=2),
        num3("Outlier L", extra.get("outlier_loss_ratio", float("nan")), dp=2),
        num3("Skew", base.get("skew", float("nan")), dp=2),
        num3("Kurtosis", base.get("kurtosis", float("nan")), dp=2),
        num3("Ulcer", base.get("ulcer", float("nan")), dp=2),
        num3("Stability", extra.get("stability_of_timeseries", float("nan")), dp=3),
        num3("Autocorr", extra.get("autocorr_lag1", float("nan")), dp=3),
        int_("DD duration (d)", base.get("max_dd_duration_days", 0)),
    ]))
    groups.append(("Probabilistic", [
        num3("PSR", extra.get("probabilistic_sharpe", float("nan")), dp=3),
        num3("P-Sortino", extra.get("probabilistic_sortino", float("nan")), dp=3),
        num3("P-AdjSortino", extra.get("probabilistic_adj_sortino", float("nan")), dp=3),
    ]))
    groups.append(("Trade proxies", [
        pct("Win rate", extra.get("win_rate", float("nan")), sign=False),
        pct("Hit rate", base.get("hit_rate", float("nan")), sign=False),
        num3("Profit Factor", extra.get("profit_factor", float("nan")), dp=2),
        num3("Profit Ratio", extra.get("profit_ratio", float("nan")), dp=3),
        num3("Kelly", extra.get("kelly_criterion", float("nan")), dp=3),
        num3("Risk of Ruin", extra.get("risk_of_ruin", float("nan")), dp=4),
        int_("Cons. Wins", extra.get("consecutive_wins", 0)),
        int_("Cons. Losses", extra.get("consecutive_losses", 0)),
    ]))
    groups.append(("Vs Benchmark", [
        pct("Alpha (ann)", vs_b.get("alpha_ann", float("nan"))),
        num3("Beta", vs_b.get("beta", float("nan")), dp=3),
        num3("IR vs SPY", vs_b.get("ir_vs_spy", float("nan"))),
        pct("Tracking err", vs_b.get("tracking_error", float("nan")), sign=False),
        num3("Correlation", vs_b.get("corr", float("nan")), dp=3),
        num3("Up capture", vs_b.get("up_capture", float("nan")), dp=2),
        num3("Down capture", vs_b.get("down_capture", float("nan")), dp=2),
        pct("Excess CAGR", vs_b.get("excess_cagr", float("nan"))),
        num3("Treynor", extra_b.get("treynor", float("nan")), dp=3),
        num3("R²", extra_b.get("r_squared", float("nan")), dp=3),
    ]))

    blocks = []
    for title, cards in groups:
        block = (
            f"<div class='section'>"
            f"<div class='section-header'><h2 class='section-title'>"
            f"<span class='idx'>§</span>{title}</h2>"
            f"<span class='section-note'>{len(cards)} metrics</span></div>"
            f"<div class='metrics-grid'>"
            + "".join(_metric_card(*c) for c in cards)
            + "</div></div>"
        )
        blocks.append(block)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Per-scheme figure + fragment builder
# ---------------------------------------------------------------------------


# Stable div_ids used by the JS picker to find chart containers.
_CHART_DIVS = [
    "fig-snapshot", "fig-logeq", "fig-heat", "fig-annual",          # Tab 2
    "fig-uw", "fig-dd-hl",                                          # Tab 3
    "fig-rs", "fig-rso", "fig-rv", "fig-rb",                        # Tab 4
    "fig-hist", "fig-box", "fig-kde",                               # Tab 5
    "fig-reg", "fig-cumxs", "fig-rcorr", "fig-cap", "fig-decile",   # Tab 6
    "fig-cone", "fig-pbox", "fig-mc",                               # Tab 7
    "fig-stress",                                                   # Tab 8
    "fig-rt-life", "fig-pnl-dist", "fig-hold", "fig-exit-rsn",      # Tab 9 Trades
    "fig-mae-mfe", "fig-sym-pnl",                                   # Tab 9 cont.
]

# div_id → owning tab-content id. Drives lazy-render: when the picker
# changes scheme, only divs in the currently-active tab are re-painted;
# others get marked stale and repaint on tab activation. Without this,
# 50-scheme reports freeze the browser on switch (14 Plotly.react calls
# per switch × every chart, including hidden tabs).
_DIV_TO_TAB = {
    "fig-snapshot": "t_quantstats", "fig-logeq": "t_quantstats",
    "fig-heat": "t_quantstats", "fig-annual": "t_quantstats",
    "fig-uw": "t_risk", "fig-dd-hl": "t_risk",
    "fig-rs": "t_rolling", "fig-rso": "t_rolling",
    "fig-rv": "t_rolling", "fig-rb": "t_rolling",
    "fig-hist": "t_distribution", "fig-box": "t_distribution",
    "fig-kde": "t_distribution",
    "fig-reg": "t_benchmark", "fig-cumxs": "t_benchmark",
    "fig-rcorr": "t_benchmark", "fig-cap": "t_benchmark",
    "fig-decile": "t_benchmark",
    "fig-cone": "t_bootstrap", "fig-pbox": "t_bootstrap",
    "fig-mc": "t_bootstrap",
    "fig-stress": "t_stress",
    "fig-rt-life": "t_trades", "fig-pnl-dist": "t_trades",
    "fig-hold": "t_trades", "fig-exit-rsn": "t_trades",
    "fig-mae-mfe": "t_trades", "fig-sym-pnl": "t_trades",
}
assert set(_DIV_TO_TAB) == set(_CHART_DIVS), (
    "DIV_TO_TAB and _CHART_DIVS drifted — keep in sync"
)


def _build_scheme_payload(
    r: SchemeResult, spy_eq: pd.Series | None, oos_start: pd.Timestamp,
    seed_nav: float, n_tickers: int, n_total_schemes: int,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build (figures_json, html_fragments) for a single scheme."""
    base_oos = r.metrics_oos
    eq_oos = r.equity.loc[r.equity.index >= oos_start]
    ext_oos = extended_metrics(eq_oos)
    vs_oos = r.vs_spy_oos
    spy_oos_slice = (
        spy_eq.loc[spy_eq.index >= oos_start] if spy_eq is not None else None
    )
    ext_vs = extended_vs_benchmark(eq_oos, spy_oos_slice)
    label = r.scheme

    # Trade-level DataFrame — empty if walker didn't populate state.trades
    # (e.g. canary using MagicMock or strategy emits equity-only output).
    trades_df = trades_to_dataframe(r.state.trades)

    # ---- Plotly figures (per-scheme) ----
    figs = {
        "fig-snapshot": snapshot_3panel_figure(r.equity, label),
        "fig-logeq": log_equity_figure(r.equity, spy_eq, label),
        "fig-heat": monthly_heatmap_figure(r.equity, f"Monthly returns — {label}"),
        "fig-annual": annual_returns_bar_figure(r.equity, spy_eq, label),
        "fig-uw": underwater_figure(r.equity, spy_eq, label),
        "fig-dd-hl": top_drawdowns_highlight_figure(r.equity, label),
        "fig-rs": rolling_sharpe_figure(r.equity, spy_eq, label),
        "fig-rso": rolling_sortino_figure(r.equity, spy_eq, label),
        "fig-rv": rolling_volatility_figure(r.equity, spy_eq, label),
        "fig-rb": rolling_beta_figure(r.equity, spy_eq, label),
        "fig-hist": returns_histogram_figure(r.equity, "Return histograms"),
        "fig-box": returns_boxplot_figure(r.equity, "Return quantile boxplots"),
        "fig-kde": kde_vs_normal_figure(r.equity, "Daily-return KDE vs Normal"),
        "fig-reg": regression_scatter_figure(r.equity, spy_eq, label)
        if spy_eq is not None else None,
        "fig-cumxs": cumulative_excess_figure(r.equity, spy_eq, label)
        if spy_eq is not None else None,
        "fig-rcorr": rolling_correlation_figure(r.equity, spy_eq, label)
        if spy_eq is not None else None,
        "fig-cap": up_down_capture_figure(r.equity, spy_eq, label)
        if spy_eq is not None else None,
        "fig-decile": spy_decile_returns_figure(r.equity, spy_eq, label)
        if spy_eq is not None else None,
        "fig-cone": forecast_cone_figure(r.equity, label),
        "fig-pbox": perf_stats_box_figure(
            r.equity, "Bootstrap perf-stats — 2000 resamples",
        ),
        "fig-mc": monte_carlo_paths_figure(r.equity, label),
        "fig-stress": stress_grid_figure(r.equity, spy_eq, label),
        "fig-rt-life": round_trip_lifetimes_figure(trades_df),
        "fig-pnl-dist": pnl_distribution_figure(trades_df),
        "fig-hold": hold_time_distribution_figure(trades_df),
        "fig-exit-rsn": exit_reason_breakdown_figure(trades_df),
        "fig-mae-mfe": mae_mfe_scatter_figure(trades_df),
        "fig-sym-pnl": per_symbol_pnl_bar_figure(trades_df),
    }
    figures_json: dict[str, dict] = {
        k: pio.to_json(fig, validate=False) for k, fig in figs.items()
        if fig is not None
    }

    # ---- HTML fragments (per-scheme) ----
    kpi_html = _kpi_block(
        label, base_oos, ext_oos, vs_oos, n_total_schemes, seed_nav, n_tickers,
    )
    metrics_html = _metrics_block(base_oos, ext_oos, vs_oos, ext_vs)
    dd_table = drawdown_table_html(r.equity)
    stress_tbl = stress_table_html(r.equity, spy_eq)
    if r.state.daily_snapshot:
        # Strip rule-tag prefix (e.g. "seq__roll_calmar_cap20" → "roll_calmar_cap20")
        # before passing to scheme_per_name_cap, which only knows unprefixed names.
        cap = scheme_per_name_cap(_strip_rule_tag(label), DEFAULT_PER_NAME_CAP)
        cash_mob = cash_mobilization_overlay(
            r, cohort_size=n_tickers, per_name_cap=cap,
            div_id=f"fig-cash-{label}",
        )
    else:
        cash_mob = "<p style='color:var(--muted)'>(no snapshot data)</p>"

    trades_kpi = trades_kpi_strip_html(trades_df, seed_nav=seed_nav)
    trades_table = trades_summary_table_html(trades_df)
    trades_full = trades_full_table_html(trades_df)
    fragments = {
        "kpi": kpi_html,
        "metrics": metrics_html,
        "dd_table": dd_table,
        "stress_table": stress_tbl,
        "cash_mob": cash_mob,
        "trades_kpi": trades_kpi,
        "trades_table": trades_table,
        "trades_full": trades_full,
    }
    return figures_json, fragments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_mega_html(
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
    bh_label: str | None = None,
    top_n_picker: int | None = DEFAULT_TOP_N_PICKER,
    exclude_cf_family: bool = True,
    dedup_duplicates: bool = True,
    ir_floor: float = 0.0,
) -> None:
    """Render the 10-tab mega report with a scheme picker.

    ``top_n_picker`` controls how many schemes the per-scheme picker
    exposes (KPI cards + charts + trades tab). Set to ``None`` to render
    ALL schemes (ref-doc c73, 2026-05-18 — "S5 should sweep all schemes
    and surface everything in v2"). File size scales linearly: ~10 MB
    per scheme, so 48 schemes ≈ 480 MB HTML. The comparison table on
    Tab-1 always lists all schemes regardless of picker size.

    ``exclude_cf_family`` (default True, added 2026-05-18) drops every
    scheme prefixed ``cf__`` from the report. The ``cash_fraction``
    (uncapped) sizing rule is a comparison baseline — not a deployment
    candidate — and its max_entry hits the 95% buffer ceiling on every
    single-signal bar, producing dozens of synonyms-of-95% rows that
    dominate the picker. Pass False to surface them for audit.

    ``dedup_duplicates`` (default True, added 2026-05-18) collapses
    schemes whose OOS equity curves hash-match (rounded to 6 sig figs
    of relative precision). Representative picked by tiebreaker:
    deployed > simpler-name > lowest-cap > alpha. Aliases listed in
    the ``≡`` column of the comparison table.
    """
    if summary.empty or not results:
        out_path.write_text(
            f"<html><body><p>{strategy_label}: no results</p></body></html>",
        )
        return

    n_in = len(summary)
    summary, results, _aliases = _filter_and_dedup(
        summary, results, oos_start, deployed_scheme,
        exclude_cf_family=exclude_cf_family,
        dedup_duplicates=dedup_duplicates,
    )
    n_out = len(summary)
    if n_out < n_in:
        n_collapsed = sum(len(v) for v in _aliases.values())
        n_cf_dropped = n_in - n_out - n_collapsed
        print(
            f"[report_v2] cohort filter: {n_in} → {n_out} "
            f"({n_cf_dropped} cf__ dropped, "
            f"{n_collapsed} alias rows collapsed)",
        )

    summary_sorted = summary.sort_values(
        "oos_sharpe", ascending=False, na_position="last",
    ).reset_index(drop=True)
    top3 = set(summary_sorted["scheme"].head(3).tolist())

    # --- Beat-SPY composite filter for the picker (2026-05-18, ref-doc c1) ---
    # The full picker (top_n_picker entries) drives file size — each picked
    # scheme adds ~2 MB. Pre-filter to schemes that beat SPY on a 3-criterion
    # composite (OOS Sharpe > SPY, IR vs SPY > 0, OOS end_value > SPY end_value).
    # Fall back to top-N by OOS Sharpe if zero schemes pass.
    spy_oos_sr = float("nan")
    spy_oos_end = float("nan")
    picker_banner: str = ""
    if spy_eq is not None and len(spy_eq) > 0:
        spy_oos = spy_eq.loc[spy_eq.index >= oos_start]
        if len(spy_oos) >= 30 and float(spy_oos.iloc[0]) > 0:
            spy_oos_rebased = (
                spy_oos / float(spy_oos.iloc[0]) * seed_nav
            )
            spy_metrics = equity_metrics(spy_oos_rebased)
            spy_oos_sr = float(spy_metrics.get("sharpe", float("nan")))
            spy_oos_end = float(spy_metrics.get("end_value", float("nan")))

    def _beats_spy(row: pd.Series) -> bool:
        if not (pd.notna(spy_oos_sr) and pd.notna(spy_oos_end)):
            return False  # SPY unknown → defer to fallback path
        sr = row.get("oos_sharpe")
        ir = row.get("ir_vs_spy")
        ev = row.get("oos_end_value")
        if pd.isna(sr) or pd.isna(ir) or pd.isna(ev):
            return False
        return (float(sr) > spy_oos_sr
                and float(ir) > 0.0
                and float(ev) > spy_oos_end)

    n_total_results = len(summary_sorted)
    cap = top_n_picker if top_n_picker is not None else n_total_results
    if pd.notna(spy_oos_sr) and pd.notna(spy_oos_end):
        beats_mask = summary_sorted.apply(_beats_spy, axis=1)
        n_pass = int(beats_mask.sum())
        if n_pass == 0:
            top_pick = summary_sorted["scheme"].head(cap).tolist()
            picker_banner = (
                f"⚠️ no schemes beat SPY on the composite "
                f"(OOS Sharpe > {spy_oos_sr:.3f}, IR > 0, "
                f"OOS end-value > ${spy_oos_end:,.0f}) — "
                f"picker showing top-{len(top_pick)} by OOS Sharpe as fallback."
            )
        else:
            beat_schemes = summary_sorted.loc[beats_mask, "scheme"].tolist()
            top_pick = beat_schemes[:cap]
            picker_banner = (
                f"✓ picker filtered to {len(top_pick)} of {n_pass} schemes "
                f"that beat SPY (OOS Sharpe > {spy_oos_sr:.3f}, IR > 0, "
                f"OOS end-value > ${spy_oos_end:,.0f}); capped at "
                f"top_n_picker={cap}."
            )
    else:
        # SPY unavailable — no filter applied
        top_pick = summary_sorted["scheme"].head(cap).tolist()
        picker_banner = (
            f"ⓘ SPY benchmark unavailable — picker showing top-{len(top_pick)} "
            f"by OOS Sharpe (no beat-SPY filter applied)."
        )
    by_name = {r.scheme: r for r in results}

    # ---- Per-scheme payloads (figures + HTML fragments) ----
    scheme_figs: dict[str, dict[str, str]] = {}  # {scheme: {div_id: fig_json}}
    scheme_frags: dict[str, dict[str, str]] = {}  # {scheme: {key: html}}
    picker_options: list[str] = []
    for name in top_pick:
        r = by_name.get(name)
        if r is None:
            continue
        figs_json, frags = _build_scheme_payload(
            r, spy_eq, oos_start, seed_nav, n_tickers,
            n_total_schemes=len(summary_sorted),
        )
        scheme_figs[name] = figs_json
        scheme_frags[name] = frags
        picker_options.append(name)

    if not picker_options:
        out_path.write_text(
            f"<html><body><p>{strategy_label}: no valid schemes</p></body></html>",
        )
        return

    default_scheme = picker_options[0]

    # ---- Comparison table + verdict (scheme-independent) ----
    verdict_block = verdict_html(summary_sorted, n_tickers)
    core_cols = [c for c in _CORE_COLS if c in summary_sorted.columns]
    compact = summary_sorted[core_cols]
    spy_row_full = spy_summary_row(
        spy_eq, oos_start, list(summary_sorted.columns), seed_nav=seed_nav,
    )
    spy_row_core = (
        {c: spy_row_full.get(c) for c in core_cols}
        if spy_row_full is not None else None
    )
    equity_by_scheme = {r.scheme: r.equity for r in results}
    table_html = _compact_table_html(
        compact, top3, spy_row_core, equity_by_scheme, spy_eq, oos_start,
        ir_floor=ir_floor,
    )

    if bh_label is None:
        bh_label = f"{n_tickers}-ticker buy-hold"

    # ---- Tab 1 multi-scheme overlays (scheme-independent) ----
    eq_full = equity_overlay(
        results, top_pick, spy_eq, bh_eq, oos_start,
        f"Equity — full period — top {len(top_pick)} by OOS Sharpe + SPY + {bh_label}",
        bh_label=bh_label, div_id="fig-eq-full-v2",
    )
    eq_oos = equity_overlay(
        results, top_pick, spy_eq, bh_eq, oos_start,
        f"Equity — OOS only ({oos_start.date()}+, rebased ${seed_nav:,.0f})",
        bh_label=bh_label, window_start=oos_start, rebase_to=seed_nav,
        div_id="fig-eq-oos-v2",
    )
    dd_full = drawdown_overlay(
        results, top_pick, spy_eq,
        f"Drawdown — full period — top {len(top_pick)} by OOS Sharpe vs SPY",
        div_id="fig-dd-full-v2",
    )

    # ---- Cohort tab (scheme-independent) ----
    cohort_corr = scheme_correlation_heatmap_html(
        equity_by_scheme,
        "Cross-scheme daily return correlation "
        "(SPY + Buy-Hold + top 30 by OOS Sharpe)",
        "fig-cohort-corr-v2",
        benchmark_eq=spy_eq, bh_eq=bh_eq,
    )

    # ---- Helper to wrap a fragment as a per-scheme div block ----
    def _frag_block(key: str) -> str:
        parts = []
        for scheme in picker_options:
            hidden = "" if scheme == default_scheme else "display:none"
            parts.append(
                f"<div class='scheme-frag' data-scheme='{scheme}' "
                f"data-frag='{key}' style='{hidden}'>"
                f"{scheme_frags[scheme][key]}</div>"
            )
        return "\n".join(parts)

    # ---- Lexicon block (2026-05-18, ref-doc) — collapsible glossary for
    # the new diagnostic columns. Adds ~3 KB; helps readers interpret HHI /
    # ENB / Cap % without leaving the report. ----
    lexicon_block = """
    <details style="margin:0.8rem 0; padding:0.6rem 0.9rem;
                    background:rgba(63,185,80,0.04);
                    border-left:3px solid var(--accent);
                    border-radius:var(--radius-sm); font-size:0.9rem;">
      <summary style="cursor:pointer; font-weight:600; color:var(--accent);">
        Metric lexicon &mdash; concentration &amp; cap-diagnostic columns
      </summary>
      <table style="width:100%; margin-top:0.6rem; font-size:0.85rem;
                    border-collapse:collapse;">
        <thead><tr style="border-bottom:1px solid var(--border);">
          <th style="text-align:left; padding:0.3rem 0.5rem;">Column</th>
          <th style="text-align:left; padding:0.3rem 0.5rem;">Definition</th>
          <th style="text-align:left; padding:0.3rem 0.5rem;">How to read it</th>
        </tr></thead><tbody>
        <tr><td style="padding:0.3rem 0.5rem;"><code>max HHI</code></td>
          <td style="padding:0.3rem 0.5rem;">Peak Herfindahl-Hirschman Index across all bars: <code>&Sigma; w<sub>i</sub>&sup2;</code>. Range [1/N, 1].</td>
          <td style="padding:0.3rem 0.5rem;">1.0 = single-name concentration on the worst bar. 1/N = perfectly diversified. Higher = more concentrated.</td></tr>
        <tr><td style="padding:0.3rem 0.5rem;"><code>ENB</code></td>
          <td style="padding:0.3rem 0.5rem;">Effective Number of Bets &mdash; Meucci 2009 lower bound, <code>1/HHI</code>. Time-averaged across active bars.</td>
          <td style="padding:0.3rem 0.5rem;">"Behaves like ENB equally-weighted independent positions." Lower = more concentrated. ENB &lt; 3 on a 50-name cohort is a red flag.</td></tr>
        <tr><td style="padding:0.3rem 0.5rem;"><code>Cap %</code></td>
          <td style="padding:0.3rem 0.5rem;"><code>n_cap_bound / n_entries_total</code> &mdash; fraction of entries where the per-name cap clamped target_$.</td>
          <td style="padding:0.3rem 0.5rem;">0% = cap never matters (you could remove it). 100% = cap is the sizing mechanism. Useful for sizing the cap.</td></tr>
        <tr><td style="padding:0.3rem 0.5rem;"><code>p95 ent%</code></td>
          <td style="padding:0.3rem 0.5rem;">95th-percentile per-entry size as % of free cash at the entry bar.</td>
          <td style="padding:0.3rem 0.5rem;">Concentration tail. High = a few entries are unusually large relative to free cash.</td></tr>
        <tr><td style="padding:0.3rem 0.5rem;"><code>max ent%</code></td>
          <td style="padding:0.3rem 0.5rem;">Largest single entry size as % of free cash.</td>
          <td style="padding:0.3rem 0.5rem;">Concentration peak. For capped sizing rules this should &le; cap &times; 100 &times; buffer (e.g., 19% for cap=20%).</td></tr>
        <tr><td style="padding:0.3rem 0.5rem;"><code>TooSm</code></td>
          <td style="padding:0.3rem 0.5rem;">Entries skipped because <code>target_$ &lt; cost_per_share</code>.</td>
          <td style="padding:0.3rem 0.5rem;">Sizing-rule pathology indicator. High count = the rule wastes signals (free cash too small to buy 1 share).</td></tr>
      </tbody></table>
      <p style="margin-top:0.6rem; font-size:0.8rem; color:var(--text-muted);">
        Sizing-rule legend: <code>cf</code> = cash_fraction,
        <code>cfc</code> = cash_fraction_capped (snapshot cap),
        <code>seq</code> = cash_fraction_seq_capped (sequential cash consumption),
        <code>plg</code> = paleologo_strict (snapshot cap + optional DD throttle).
        Scheme picker keys are prefixed with these tags (e.g., <code>seq__roll_calmar_cap20</code>).
      </p>
    </details>
    """

    # ---- Picker banner: shows whether beat-SPY filter is active/fell back ----
    picker_banner_block = (
        f"<div style='margin:0.6rem 0; padding:0.5rem 0.8rem; "
        f"background:rgba(99,196,124,0.06); "
        f"border-left:3px solid var(--accent); "
        f"border-radius:var(--radius-sm); font-size:0.85rem; "
        f"color:var(--text);'>{picker_banner}</div>"
        if picker_banner else ""
    )

    # ---- Tab content builders ----
    tab1 = f"""
    <div class='section'>{_frag_block("kpi")}</div>
    <div class='section'>{verdict_block}</div>
    {picker_banner_block}
    {lexicon_block}
    <div class='section'>
      <div class='section-header'><h2 class='section-title'>
      <span class='idx'>§</span>Scheme comparison</h2>
      <span class='section-note'>{len(summary_sorted)} schemes · sortable · SPY pinned</span></div>
      {table_html}
    </div>
    <div class='section'>
      <div class='section-header'><h2 class='section-title'>
      <span class='idx'>§</span>Equity — full period</h2></div>
      <div class='chart-full'><div class='chart-body'>{eq_full}</div></div>
    </div>
    <div class='section'>
      <div class='section-header'><h2 class='section-title'>
      <span class='idx'>§</span>Equity — OOS only (rebased)</h2></div>
      <div class='chart-full'><div class='chart-body'>{eq_oos}</div></div>
    </div>
    <div class='section'>
      <div class='section-header'><h2 class='section-title'>
      <span class='idx'>§</span>Drawdown — full period</h2></div>
      <div class='chart-full'><div class='chart-body'>{dd_full}</div></div>
    </div>
    <div class='section'>{_frag_block("metrics")}</div>
    """

    def _chart_section(title: str, div_id: str) -> str:
        return (
            f"<div class='section'>"
            f"<div class='section-header'><h2 class='section-title'>"
            f"<span class='idx'>§</span>{title}</h2></div>"
            f"<div class='chart-full'>"
            f"<div class='chart-body'><div id='{div_id}' "
            f"style='width:100%;min-height:340px'></div></div></div></div>"
        )

    tab2 = (
        _chart_section("Snapshot — 2-panel", "fig-snapshot")
        + _chart_section("Log-scale cumulative return", "fig-logeq")
        + _chart_section("Monthly returns heatmap", "fig-heat")
        + _chart_section("Annual returns", "fig-annual")
    )
    tab3 = (
        _chart_section("Underwater — % below peak", "fig-uw")
        + _chart_section("Top-5 drawdowns highlighted", "fig-dd-hl")
        + f"<div class='section'><div class='section-header'>"
          f"<h2 class='section-title'><span class='idx'>§</span>"
          f"Drawdown periods — peak → valley → recovery</h2></div>"
          f"{_frag_block('dd_table')}</div>"
    )
    tab4 = """
    <div class='chart-row'>
      <div class='chart-panel'><div class='chart-body'>
      <div id='fig-rs' style='width:100%;min-height:320px'></div></div></div>
      <div class='chart-panel'><div class='chart-body'>
      <div id='fig-rso' style='width:100%;min-height:320px'></div></div></div>
    </div>
    <div class='chart-row'>
      <div class='chart-panel'><div class='chart-body'>
      <div id='fig-rv' style='width:100%;min-height:320px'></div></div></div>
      <div class='chart-panel'><div class='chart-body'>
      <div id='fig-rb' style='width:100%;min-height:320px'></div></div></div>
    </div>
    """
    tab5 = (
        _chart_section("Return histograms (D / W / M)", "fig-hist")
        + _chart_section("Return quantiles — boxplots", "fig-box")
        + _chart_section("Daily-return KDE vs Normal", "fig-kde")
    )
    tab6 = (
        _chart_section("Cumulative return — strategy + SPY + excess", "fig-cumxs")
        + _chart_section("Regression vs SPY (α / β / R²)", "fig-reg")
        + _chart_section("Rolling 252-bar correlation vs SPY", "fig-rcorr")
        + "<div class='chart-row'>"
          "<div class='chart-panel'><div class='chart-body'>"
          "<div id='fig-cap' style='width:100%;min-height:380px'></div>"
          "</div></div>"
          "<div class='chart-panel'><div class='chart-body'>"
          "<div id='fig-decile' style='width:100%;min-height:380px'></div>"
          "</div></div></div>"
    )
    tab7 = (
        _chart_section("Forecast cone — 5000 bootstrap paths", "fig-cone")
        + _chart_section("Bootstrap perf-stats — 2000 resamples", "fig-pbox")
        + _chart_section("Monte Carlo simulation", "fig-mc")
    )
    tab8 = (
        _chart_section("Interesting times — 25 stress windows", "fig-stress")
        + f"<div class='section'>{_frag_block('stress_table')}</div>"
    )
    tab9 = (
        f"<div class='section'>{_frag_block('trades_kpi')}</div>"
        + f"<div class='section'><div class='section-header'>"
          f"<h2 class='section-title'><span class='idx'>§</span>"
          f"Per exit-reason summary</h2></div>"
          f"{_frag_block('trades_table')}</div>"
        + _chart_section("Round-trip lifetimes (size = |$ PnL|, color = win/loss)", "fig-rt-life")
        + "<div class='chart-row'>"
        + "<div class='chart-panel'><div class='chart-body'>"
        + "<div id='fig-pnl-dist' style='width:100%;min-height:380px'></div>"
        + "</div></div>"
        + "<div class='chart-panel'><div class='chart-body'>"
        + "<div id='fig-hold' style='width:100%;min-height:380px'></div>"
        + "</div></div></div>"
        + "<div class='chart-row'>"
        + "<div class='chart-panel'><div class='chart-body'>"
        + "<div id='fig-exit-rsn' style='width:100%;min-height:380px'></div>"
        + "</div></div>"
        + "<div class='chart-panel'><div class='chart-body'>"
        + "<div id='fig-mae-mfe' style='width:100%;min-height:460px'></div>"
        + "</div></div></div>"
        + _chart_section("Per-symbol PnL (top + bottom 25)", "fig-sym-pnl")
        + f"<div class='section'><div class='section-header'>"
          f"<h2 class='section-title'><span class='idx'>§</span>"
          f"All individual trades</h2>"
          f"<span class='section-note'>sortable headers · type to filter</span>"
          f"</div>{_frag_block('trades_full')}</div>"
    )
    tab10 = f"""
    <div class='section'>
      <div class='section-header'><h2 class='section-title'>
      <span class='idx'>§</span>Cash mobilization — 4-panel</h2></div>
      {_frag_block("cash_mob")}
    </div>
    """
    tab11 = f"""
    <div class='section'>
      <div class='chart-full'><div class='chart-body'>{cohort_corr}</div></div>
    </div>
    """

    # ---- Picker JS payload ----
    figs_payload = {
        div_id: {sch: scheme_figs[sch][div_id] for sch in picker_options
                 if div_id in scheme_figs[sch]}
        for div_id in _CHART_DIVS
    }
    figs_payload_json = (
        "{"
        + ",".join(
            f'"{div_id}":{{'
            + ",".join(f'"{sch}":{fig_json}' for sch, fig_json in by_sch.items())
            + "}"
            for div_id, by_sch in figs_payload.items()
        )
        + "}"
    )
    options_json = json.dumps(picker_options)
    default_json = json.dumps(default_scheme)
    div_to_tab_json = json.dumps(_DIV_TO_TAB)

    css, js = _inline_assets()
    deployed_label = deployed_scheme if deployed_scheme is not None else "(none)"
    html_out = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <title>{strategy_label} — S5 v2 mega-report (picker)</title>
  <style>{css}</style>
  <style>
    .picker-bar {{
      display: flex; align-items: center; gap: 12px;
      padding: 10px 14px; background: var(--bg-elev, #161b22);
      border: 1px solid var(--border, #30363d); border-radius: 6px;
      margin: 10px 0 16px 0; position: sticky; top: 0; z-index: 50;
    }}
    .picker-bar label {{ color: var(--muted, #8b949e); font-size: 12px;
      text-transform: uppercase; letter-spacing: 0.08em; }}
    .picker-bar select {{
      background: var(--bg-card, #0d1117); color: var(--text, #e6edf3);
      border: 1px solid var(--border, #30363d); padding: 6px 10px;
      border-radius: 4px; font-family: var(--font-mono, monospace);
      font-size: 13px; min-width: 220px;
    }}
    .picker-bar .selected-scheme-label {{
      color: var(--accent, #58a6ff); font-weight: 700;
      font-family: var(--font-mono); font-size: 13px;
    }}
  </style>
</head>
<body>
<div class="page">

  <header class="report-header">
    <div>
      <h1><span class="accent-dot"></span>{strategy_label} — S5 v2 mega-report</h1>
      <div class="meta">
        <span><span class="label">Generated</span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        <span class="sep">·</span>
        <span><span class="label">Sizing</span><code>{sizing_rule}</code></span>
        <span class="sep">·</span>
        <span><span class="label">Cohort</span>{n_tickers} tickers</span>
        <span class="sep">·</span>
        <span><span class="label">OOS start</span>{oos_start.date()}</span>
        <span class="sep">·</span>
        <span><span class="label">Seed NAV</span>${seed_nav:,.0f}</span>
        <span class="sep">·</span>
        <span><span class="label">Deployed</span>{deployed_label}</span>
      </div>
    </div>
    <div class="controls">
      <button class="icon-btn" id="theme-toggle"><span class="theme-label">Dark</span></button>
      <button class="icon-btn" id="cfg-open">Config <kbd>⌥C</kbd></button>
      <button class="icon-btn" onclick="window.print()">Print</button>
    </div>
  </header>

  <div class="picker-bar">
    <label>Selected scheme</label>
    <select id="scheme-picker">
      {''.join(f'<option value="{s}">{s}</option>' for s in picker_options)}
    </select>
    <span style="color:var(--muted);font-size:11px">
      top {len(picker_options)} by OOS Sharpe ·
      switches KPI cards, metric tiles, and every chart in tabs 2-9
    </span>
    <span style="flex:1"></span>
    <span style="color:var(--muted);font-size:11px">currently viewing:</span>
    <span class="selected-scheme-label">{default_scheme}</span>
  </div>

  <nav class="tabs">
    <button class="tab-btn active" data-tab="t_overview">Overview</button>
    <button class="tab-btn" data-tab="t_quantstats">Quantstats</button>
    <button class="tab-btn" data-tab="t_risk">Risk</button>
    <button class="tab-btn" data-tab="t_rolling">Rolling</button>
    <button class="tab-btn" data-tab="t_distribution">Distribution</button>
    <button class="tab-btn" data-tab="t_benchmark">Benchmark</button>
    <button class="tab-btn" data-tab="t_bootstrap">Bootstrap</button>
    <button class="tab-btn" data-tab="t_stress">Stress <span class="tab-count">25</span></button>
    <button class="tab-btn" data-tab="t_trades">Trades</button>
    <button class="tab-btn" data-tab="t_execution">Execution</button>
    <button class="tab-btn" data-tab="t_cohort">Cohort</button>
  </nav>

  <section id="t_overview" class="tab-content active">{tab1}</section>
  <section id="t_quantstats" class="tab-content">{tab2}</section>
  <section id="t_risk" class="tab-content">{tab3}</section>
  <section id="t_rolling" class="tab-content">{tab4}</section>
  <section id="t_distribution" class="tab-content">{tab5}</section>
  <section id="t_benchmark" class="tab-content">{tab6}</section>
  <section id="t_bootstrap" class="tab-content">{tab7}</section>
  <section id="t_stress" class="tab-content">{tab8}</section>
  <section id="t_trades" class="tab-content">{tab9}</section>
  <section id="t_execution" class="tab-content">{tab10}</section>
  <section id="t_cohort" class="tab-content">{tab11}</section>
</div>

<aside id="cfg-drawer" class="cfg-drawer">
  <h3>Configuration</h3>
  <div class="cfg-sub">Live-edit CSS tokens · <kbd>Alt+C</kbd></div>
  <div class="cfg-list"></div>
  <div class="cfg-actions">
    <button class="cfg-reset">Reset</button>
    <button class="cfg-copy primary">Copy CSS</button>
  </div>
</aside>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>{js}</script>
<script>
  const SCHEME_FIGS = {figs_payload_json};
  const SCHEME_OPTIONS = {options_json};
  const DIV_TO_TAB = {div_to_tab_json};
  let CURRENT_SCHEME = {default_json};

  // Inverse map: tab_id → [div_id, ...]. Cached once.
  const TAB_DIVS = (function () {{
    const m = {{}};
    Object.keys(DIV_TO_TAB).forEach(function (divId) {{
      const tab = DIV_TO_TAB[divId];
      (m[tab] = m[tab] || []).push(divId);
    }});
    return m;
  }})();

  // PAINTED[divId] = scheme name currently drawn into that div.
  // undefined → never painted; '__stale__' → wrong scheme, needs repaint.
  const PAINTED = {{}};

  function paintChart(divId) {{
    const figStr = (SCHEME_FIGS[divId] || {{}})[CURRENT_SCHEME];
    const el = document.getElementById(divId);
    if (!el || !figStr) return;
    if (PAINTED[divId] === CURRENT_SCHEME) return;
    let fig;
    try {{ fig = (typeof figStr === 'string') ? JSON.parse(figStr) : figStr; }}
    catch (e) {{ console.error('parse fail', divId, e); return; }}
    Plotly.react(divId, fig.data || [], fig.layout || {{}},
                 {{responsive: true, displaylogo: false}});
    PAINTED[divId] = CURRENT_SCHEME;
  }}

  function renderTab(tabId) {{
    const divs = TAB_DIVS[tabId];
    if (!divs) return;  // Tab 1 / 10 / 11 have no picker-swappable charts
    divs.forEach(paintChart);
  }}

  function renderActiveTab() {{
    const active = document.querySelector('.tab-content.active');
    if (active) renderTab(active.id);
  }}

  function renderScheme(scheme) {{
    CURRENT_SCHEME = scheme;
    // HTML fragments — cheap, swap all at once.
    document.querySelectorAll('.scheme-frag').forEach(function (el) {{
      el.style.display = (el.dataset.scheme === scheme) ? '' : 'none';
    }});
    document.querySelectorAll('.selected-scheme-label').forEach(function (el) {{
      el.textContent = scheme;
    }});
    // Invalidate every painted div; only the active tab repaints now,
    // others repaint when activated.
    Object.keys(PAINTED).forEach(function (divId) {{
      if (PAINTED[divId] !== undefined) PAINTED[divId] = '__stale__';
    }});
    renderActiveTab();
  }}

  document.addEventListener('DOMContentLoaded', function () {{
    // Watch tab activation — repaint the now-active tab if it has stale
    // or never-painted divs. The kit's switchTab() toggles `.active` on
    // .tab-content elements; one mutation per tab per switch.
    document.querySelectorAll('.tab-content').forEach(function (el) {{
      const obs = new MutationObserver(function (muts) {{
        for (let i = 0; i < muts.length; i++) {{
          if (muts[i].attributeName === 'class'
              && el.classList.contains('active')) {{
            renderTab(el.id);
            break;
          }}
        }}
      }});
      obs.observe(el, {{attributes: true, attributeFilter: ['class']}});
    }});
    // Initial paint of whichever tab is active on load (typically t_overview,
    // which has no picker charts → zero Plotly inits at page open).
    renderScheme(CURRENT_SCHEME);
    const sel = document.getElementById('scheme-picker');
    sel.addEventListener('change', function (e) {{
      renderScheme(e.target.value);
    }});
  }});
</script>
</body></html>
"""
    out_path.write_text(html_out)


# Silence unused-import warnings for items kept available for callers
_ = (equity_metrics, vs_spy)
