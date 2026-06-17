"""Per-scheme runner + ``run_all_schemes`` aggregator.

Extracted from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` lines
~862-972 + main() loop (2026-04-30). Strategy-agnostic.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_tester.s5_replay.metrics import equity_metrics, vs_spy
from strategy_tester.s5_replay.oracles import (
    build_oracle,
    scheme_per_name_cap,
)
from strategy_tester.s5_replay.walker import (
    DEFAULT_COMMISSION_MODEL,
    DEFAULT_DD_TOL,
    DEFAULT_PER_NAME_CAP,
    DEFAULT_SIZING_RULE,
    ReplayState,
    walk_portfolio_oracle,
)

log = logging.getLogger(__name__)

# Fork-worker shared state — set in parent before Pool creation so children
# inherit it COW (no pickling of the large cache / rolling DataFrames).
_WORKER_STATE: dict = {}


def _fork_worker(scheme: str) -> SchemeResult:
    s = _WORKER_STATE
    return run_scheme(
        scheme, s["cache"], s["base_maps"], s["rolling_sr"],
        seed_nav=s["seed_nav"], buffer=s["buffer"],
        spy_eq=s["spy_eq"], oos_start=s["oos_start"],
        sizing_rule=s["sizing_rule"],
        per_name_cap_default=s["per_name_cap_default"],
        dd_throttle=s["dd_throttle"], dd_tol=s["dd_tol"],
        commission_model=s["commission_model"],
        rolling_iv=s["rolling_iv"], rolling_sortino=s["rolling_sortino"],
        rolling_calmar=s["rolling_calmar"], rolling_mom=s["rolling_mom"],
        rolling_er=s["rolling_er"],
    )


@dataclass
class SchemeResult:
    scheme: str
    equity: pd.Series
    metrics_full: dict
    metrics_oos: dict
    vs_spy_oos: dict
    n_trades: int
    n_failed: int
    n_clipped: int
    n_too_small: int
    mean_clip_ratio: float
    max_entry_pct_cash: float
    p95_entry_pct_cash: float
    max_entry_pct_nav: float
    # Concentration diagnostics (added 2026-05-18, ref-doc Q5/Q21):
    # HHI = Σ w_i² (active days only); ENB = 1/HHI (Meucci 2009 dual basis).
    # mean_* are time-averaged across active days; max_* are bar peaks.
    mean_hhi: float
    max_hhi: float
    mean_enb: float
    min_enb: float
    # Cap-binding telemetry (added 2026-05-18, ref-doc Q17 c44).
    n_cap_bound: int
    n_entries_total: int
    cap_bound_rate: float  # n_cap_bound / n_entries_total, NaN if 0 entries
    state: ReplayState


def _hhi_enb_from_snapshots(state: ReplayState) -> tuple[float, float, float, float]:
    """Compute mean/max HHI + mean/min ENB from per-bar weight vectors.

    HHI (Herfindahl-Hirschman Index) on portfolio weights: Σ w_i² summed over
    active positions only — pure cash bars (n_positions=0) are excluded so
    the diagnostic doesn't get washed out by sparse-signal periods.

    ENB (Effective Number of Bets, Meucci 2009 §2.1) = 1/HHI under the simplest
    diversification dual (assumes orthogonal bets). For correlated bets the
    full Meucci ENB requires PCA on Σ — but 1/HHI is the lower-bound
    diversification number and is the standard quick-look diagnostic.

    Returns (mean_hhi, max_hhi, mean_enb, min_enb). NaN when no active bars.
    """
    hhi_per_bar = []
    enb_per_bar = []
    for snap in state.daily_snapshot:
        w = snap.get("weights") or {}
        if not w:
            continue
        vals = np.fromiter(w.values(), dtype=np.float64)
        if vals.size == 0:
            continue
        hhi = float((vals ** 2).sum())
        if hhi <= 1e-12:
            continue
        hhi_per_bar.append(hhi)
        enb_per_bar.append(1.0 / hhi)
    if not hhi_per_bar:
        nan = float("nan")
        return nan, nan, nan, nan
    return (
        float(np.mean(hhi_per_bar)),
        float(max(hhi_per_bar)),
        float(np.mean(enb_per_bar)),
        float(min(enb_per_bar)),
    )


def run_scheme(
    scheme: str,
    cache: dict[str, dict],
    base_maps: dict[str, dict[str, float]],
    rolling_sr: pd.DataFrame | None,
    seed_nav: float,
    buffer: float,
    spy_eq: pd.Series | None,
    oos_start: pd.Timestamp,
    sizing_rule: str = DEFAULT_SIZING_RULE,
    per_name_cap_default: float = DEFAULT_PER_NAME_CAP,
    rolling_iv: pd.DataFrame | None = None,
    rolling_sortino: pd.DataFrame | None = None,
    rolling_calmar: pd.DataFrame | None = None,
    rolling_mom: pd.DataFrame | None = None,
    rolling_er: pd.DataFrame | None = None,
    dd_throttle: str = "off",
    dd_tol: float = DEFAULT_DD_TOL,
    commission_model: str = DEFAULT_COMMISSION_MODEL,
) -> SchemeResult:
    """Run a single scheme through the cash-aware walker."""
    oracle = build_oracle(
        scheme, base_maps, rolling_sr,
        rolling_iv=rolling_iv,
        rolling_sortino=rolling_sortino,
        rolling_calmar=rolling_calmar,
        rolling_mom=rolling_mom,
        rolling_er=rolling_er,
    )
    # paleologo_strict reads per-name cap from the scheme name's suffix
    # (`*_cap05` → 0.05) so the cap-sweep is encoded in the scheme list;
    # static schemes fall back to the CLI default.
    per_name_cap = scheme_per_name_cap(scheme, per_name_cap_default)
    state = walk_portfolio_oracle(
        cache, weight_oracle=oracle, seed_nav=seed_nav, buffer=buffer,
        sizing_rule=sizing_rule,
        per_name_cap=per_name_cap,
        dd_throttle=dd_throttle,
        dd_tol=dd_tol,
        commission_model=commission_model,
    )
    snap = pd.DataFrame(state.daily_snapshot)
    eq = snap.set_index("date")["netliq"]
    metrics_full = equity_metrics(eq)
    eq_oos = eq.loc[eq.index >= oos_start]
    # Rebase OOS slice to seed_nav so oos_end_value reflects OOS-only growth.
    # Without rebasing, eq_oos.iloc[-1] == eq.iloc[-1] (same final value as
    # full-period), making oos_end_value indistinguishable from full end_value.
    # Sharpe/CAGR/MaxDD are scale-invariant — they were already correct.
    # Bug surfaced 2026-05-10 (cum_rsi_v2). Fix mirrors spy_summary_row.
    if len(eq_oos) > 0 and float(eq_oos.iloc[0]) > 0:
        eq_oos_rebased = eq_oos / float(eq_oos.iloc[0]) * seed_nav
    else:
        eq_oos_rebased = eq_oos
    metrics_oos = equity_metrics(eq_oos_rebased)
    spy_oos = (
        spy_eq.loc[spy_eq.index >= oos_start] if spy_eq is not None else None
    )
    vs = vs_spy(eq_oos, spy_oos)  # vs_spy uses pct_change — scale-invariant — pass raw slice
    n_clipped = sum(1 for f in state.failed_entries if f.reason == "clipped")
    n_failed = sum(1 for f in state.failed_entries if f.reason == "no_cash")
    n_too_small = sum(1 for f in state.failed_entries if f.reason == "too_small")
    clip_rows = [
        f for f in state.failed_entries
        if f.reason == "clipped" and f.target_dollars > 0
    ]
    mean_clip_ratio = (
        float(np.mean([f.sized_dollars / f.target_dollars for f in clip_rows]))
        if clip_rows else float("nan")
    )
    # Cap-verification diagnostics: peak per-trade size as % of cash and NAV.
    # For paleologo_strict these MUST be ≤ cap × 100 (e.g., ≤ 9.5% for cap=10%
    # × buffer 0.95). For older modes they exceed cap routinely.
    entry_pct_cash_arr = [t.entry_pct_cash for t in state.trades]
    entry_pct_nav_arr = [t.entry_pct_nav for t in state.trades]
    max_entry_pct_cash = (
        float(max(entry_pct_cash_arr)) if entry_pct_cash_arr else float("nan")
    )
    max_entry_pct_nav = (
        float(max(entry_pct_nav_arr)) if entry_pct_nav_arr else float("nan")
    )
    p95_entry_pct_cash = (
        float(np.percentile(entry_pct_cash_arr, 95)) if entry_pct_cash_arr
        else float("nan")
    )
    mean_hhi, max_hhi, mean_enb, min_enb = _hhi_enb_from_snapshots(state)
    cap_bound_rate = (
        float(state.n_cap_bound / state.n_entries_total)
        if state.n_entries_total > 0 else float("nan")
    )
    return SchemeResult(
        scheme=scheme, equity=eq,
        metrics_full=metrics_full, metrics_oos=metrics_oos,
        vs_spy_oos=vs,
        n_trades=len(state.trades),
        n_failed=n_failed, n_clipped=n_clipped,
        n_too_small=n_too_small,
        mean_clip_ratio=mean_clip_ratio,
        max_entry_pct_cash=max_entry_pct_cash,
        p95_entry_pct_cash=p95_entry_pct_cash,
        max_entry_pct_nav=max_entry_pct_nav,
        mean_hhi=mean_hhi, max_hhi=max_hhi,
        mean_enb=mean_enb, min_enb=min_enb,
        n_cap_bound=state.n_cap_bound,
        n_entries_total=state.n_entries_total,
        cap_bound_rate=cap_bound_rate,
        state=state,
    )


def run_all_schemes(
    schemes_to_run: list[str],
    cache: dict[str, dict],
    base_maps: dict[str, dict[str, float]],
    rolling_sr: pd.DataFrame | None,
    seed_nav: float,
    buffer: float,
    spy_eq: pd.Series | None,
    oos_start: pd.Timestamp,
    sizing_rule: str = DEFAULT_SIZING_RULE,
    per_name_cap_default: float = DEFAULT_PER_NAME_CAP,
    dd_throttle: str = "off",
    dd_tol: float = DEFAULT_DD_TOL,
    commission_model: str = DEFAULT_COMMISSION_MODEL,
    rolling_iv: pd.DataFrame | None = None,
    rolling_sortino: pd.DataFrame | None = None,
    rolling_calmar: pd.DataFrame | None = None,
    rolling_mom: pd.DataFrame | None = None,
    rolling_er: pd.DataFrame | None = None,
    workers: int = 1,
) -> list[SchemeResult]:
    """Run all schemes, optionally in parallel.

    workers=1  → sequential (original behaviour, order preserved).
    workers>1  → mp.get_context("fork").Pool; cache/DataFrames are inherited
                 COW by child processes (no pickling), bypassing the GIL.
                 Results arrive in completion order (not scheme order).
    """
    n = len(schemes_to_run)

    if workers <= 1:
        results: list[SchemeResult] = []
        for i, scheme in enumerate(schemes_to_run, 1):
            ts = time.time()
            try:
                r = run_scheme(
                    scheme, cache, base_maps, rolling_sr,
                    seed_nav=seed_nav, buffer=buffer,
                    spy_eq=spy_eq, oos_start=oos_start,
                    sizing_rule=sizing_rule,
                    per_name_cap_default=per_name_cap_default,
                    dd_throttle=dd_throttle, dd_tol=dd_tol,
                    commission_model=commission_model,
                    rolling_iv=rolling_iv, rolling_sortino=rolling_sortino,
                    rolling_calmar=rolling_calmar, rolling_mom=rolling_mom,
                    rolling_er=rolling_er,
                )
            except Exception as exc:
                log.error("  [%d/%d] %s: FAILED — %s", i, n, scheme, exc)
                continue
            results.append(r)
            log.info(
                "  [%d/%d] %-26s OOS SR=%6.3f  CAGR=%7.2f%%  MaxDD=%6.2f%%  "
                "done in %4.1fs",
                i, n, scheme,
                r.metrics_oos["sharpe"],
                (r.metrics_oos["cagr"] or 0) * 100,
                (r.metrics_oos["max_dd"] or 0) * 100,
                time.time() - ts,
            )
        return results

    # Parallel path — fork inherits the large cache/DataFrames COW; only
    # scheme name strings are sent through the pipe per task.
    global _WORKER_STATE
    _WORKER_STATE = dict(
        cache=cache, base_maps=base_maps, rolling_sr=rolling_sr,
        seed_nav=seed_nav, buffer=buffer, spy_eq=spy_eq, oos_start=oos_start,
        sizing_rule=sizing_rule, per_name_cap_default=per_name_cap_default,
        dd_throttle=dd_throttle, dd_tol=dd_tol,
        commission_model=commission_model,
        rolling_iv=rolling_iv, rolling_sortino=rolling_sortino,
        rolling_calmar=rolling_calmar, rolling_mom=rolling_mom,
        rolling_er=rolling_er,
    )
    results_parallel: list[SchemeResult] = []
    completed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_fork_worker, schemes_to_run):
            completed += 1
            results_parallel.append(r)
            log.info(
                "  [%d/%d] %-26s OOS SR=%6.3f  CAGR=%7.2f%%  MaxDD=%6.2f%%",
                completed, n, r.scheme,
                r.metrics_oos["sharpe"],
                (r.metrics_oos["cagr"] or 0) * 100,
                (r.metrics_oos["max_dd"] or 0) * 100,
            )
    return results_parallel
