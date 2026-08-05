"""Expanding walk-forward validation. Pardo 2e §8.3.
NAME-COLLISION NOTE (2026-08-03 dedup audit): this is NOT
``lib/probe/wfa.py::walk_forward_expanding`` (retired, hard-error guarded,
77.5% selection/grading overlap at defaults). This module builds genuine
IS/OOS date folds — but note only ``window`` is re-fit per fold;
entry/exit/stop/slope thresholds come fixed from the S2 row.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.vbt_runner import (
    compute_halflife,
    compute_ratio,
    backtest_numba_fold,
)
from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    bars_per_year_from_index,
    geometric_cagr,
    max_drawdown,
)

# ── Worker globals (set once per fork via _worker_init) ──────────
_WORKER_PRICES: pd.DataFrame | None = None
_WORKER_SIGNAL_FN: Callable | None = None


def _worker_init(
    prices: pd.DataFrame,
    signal_fn: Callable | None,
) -> None:
    global _WORKER_PRICES, _WORKER_SIGNAL_FN
    _WORKER_PRICES = prices
    _WORKER_SIGNAL_FN = signal_fn


def build_expanding_folds(
    common_idx: pd.Index,
    n_folds: int = 8,
) -> list[tuple[pd.Index, pd.Index]]:
    """IS grows, OOS = fixed chunk. Pardo 2e §8.3."""
    n = len(common_idx)
    chunk = n // (n_folds + 1)
    if chunk < 20:
        return []
    folds = []
    for k in range(n_folds):
        is_end = (k + 1) * chunk
        oos_end = min((k + 2) * chunk, n)
        is_idx = common_idx[:is_end]
        oos_idx = common_idx[is_end:oos_end]
        if len(is_idx) < 50 or len(oos_idx) < 10:
            continue
        folds.append((is_idx, oos_idx))
    return folds


def _process_one_pair_wfa(
    pair_row: dict,
    n_folds: int,
    fees: float,
    slope_window: int,
    use_s2_window: bool,
    min_sharpe: float = 0.0,
    min_win_rate: float = 0.625,
    min_active_folds: int = 4,
) -> dict | None:
    """Process a single pair through expanding WFA folds.

    Runs in a forked worker — reads prices / signal_fn from module globals.
    """
    prices = _WORKER_PRICES
    signal_fn = _WORKER_SIGNAL_FN
    num = pair_row["numerator"]
    den = pair_row["denominator"]

    try:
        ratio, num_prices, common = compute_ratio(prices, pair_row)
    except KeyError:
        return None

    folds = build_expanding_folds(common, n_folds)
    if not folds:
        return None
    # Frequency-correct annualization (2026-08-05): detect bars/year from the
    # panel — the engine's bt["sharpe"] is 252-annualized regardless of bars.
    ppy = bars_per_year_from_index(common)

    fold_sharpes: list[float] = []
    is_sharpes: list[float] = []
    oos_returns_all: list[np.ndarray] = []
    for is_idx, oos_idx in folds:
        if use_s2_window:
            w = int(pair_row.get("window", 20))
        else:
            hl = compute_halflife(ratio.loc[is_idx])
            w = (
                min(max(int(hl * 0.5), 10), 252)
                if not np.isnan(hl)
                else pair_row.get("window", 20)
            )
        is_bt = backtest_numba_fold(
            num_prices.loc[is_idx],
            ratio.loc[is_idx],
            w,
            pair_row["entry_thresh"],
            pair_row["exit_thresh"],
            stop_pct=pair_row.get("stop_pct", 0.0),
            slope_min=pair_row.get("slope_min", 0.0),
            slope_window=slope_window,
            fees=fees,
            signal_fn=signal_fn,
        )
        oos_bt = backtest_numba_fold(
            num_prices.loc[oos_idx],
            ratio.loc[oos_idx],
            w,
            pair_row["entry_thresh"],
            pair_row["exit_thresh"],
            stop_pct=pair_row.get("stop_pct", 0.0),
            slope_min=pair_row.get("slope_min", 0.0),
            slope_window=slope_window,
            fees=fees,
            signal_fn=signal_fn,
        )
        oos_rets = oos_bt.get("returns")
        if oos_rets is not None:
            if isinstance(oos_rets, pd.Series):
                oos_rets = oos_rets.values
            oos_rets = np.asarray(oos_rets, dtype=np.float64)
            oos_returns_all.append(oos_rets)
        # Active-fold convention (Q6, 2026-08-04): a zero-trade OOS
        # window is absence of evidence, not a failed fold — skip it.
        # The n_active / win-rate floors below guard the
        # conditioning-on-activity bias. Mirrors lib/probe/wfa.py.
        if oos_rets is None or not np.any(oos_rets != 0):
            continue
        # SRs from fold returns via the canonical kernel at detected bars/year
        # (kernel returns 0.0 on zero-variance folds).
        fold_sharpes.append(annualized_sharpe(oos_rets, periods_per_year=ppy))
        is_rets = is_bt.get("returns")
        if isinstance(is_rets, pd.Series):
            is_rets = is_rets.values
        is_sharpes.append(
            annualized_sharpe(
                np.asarray(is_rets, dtype=np.float64), periods_per_year=ppy
            )
            if is_rets is not None and len(is_rets)
            else 0.0
        )

    # Stitch OOS returns for CAGR / MaxDD (ALL folds incl. inactive —
    # capital-efficiency view keeps idle-cash zeros)
    if oos_returns_all:
        stitched = np.concatenate(oos_returns_all)
        oos_cagr = geometric_cagr(stitched, periods_per_year=ppy)
        oos_mdd = max_drawdown(stitched)
        n_oos_bars = int(len(stitched))
    else:
        oos_cagr = 0.0
        oos_mdd = 0.0
        n_oos_bars = 0

    n_active = len(fold_sharpes)
    mean_oos = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    median_oos = float(np.median(fold_sharpes)) if fold_sharpes else 0.0
    std_oos = float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0
    mean_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    if np.isinf(mean_oos) or np.isnan(mean_oos):
        mean_oos = 0.0
    if np.isinf(mean_is) or np.isnan(mean_is):
        mean_is = 0.0
    degradation = round(1 - mean_oos / mean_is, 4) if mean_is > 1e-6 else float("nan")

    # Pardo §8.4 on ACTIVE folds: n_pos / n_active
    pct_positive = (
        float(np.sum(np.array(fold_sharpes) > 0) / n_active) if fold_sharpes else 0.0
    )

    return {
        "pair": pair_row["pair"],
        "numerator": num,
        "denominator": den,
        "mean_test_sharpe": round(mean_oos, 4),
        "median_test_sharpe": round(median_oos, 4),
        "std_test_sharpe": round(std_oos, 4),
        "n_test_periods": n_active,
        "n_active": n_active,
        "n_folds_total": len(folds),
        "n_oos_bars": n_oos_bars,
        "baseline_sharpe": round(mean_is, 4),
        "degradation": round(degradation, 4),
        "oos_cagr": round(oos_cagr, 4),
        "oos_max_dd": round(oos_mdd, 4),
        "pct_positive": round(pct_positive, 4),
        "passed": (
            mean_oos >= min_sharpe
            and pct_positive >= min_win_rate
            and n_active >= min_active_folds
        ),
        "val_method": "wfa_expanding",
        "wfe": round(mean_oos / mean_is, 4) if mean_is > 0 else 0.0,
    }


def _process_one_pair_wfa_wrapper(args: tuple) -> dict | None:
    """Unpack tuple for imap_unordered compatibility."""
    return _process_one_pair_wfa(*args)


@register_stage("s3")
def wfa_expanding(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """Expanding walk-forward: 8 folds, IS grows, OOS fixed chunk."""
    n_folds = config.get("wfa_n_folds", 8)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    min_sharpe = config.get("s3_min_oos_sharpe", 0.0)
    min_win_rate = config.get("pardo_min_active_win_rate", 0.625)
    min_active_folds = config.get("pardo_min_active_folds", 4)

    passed_pairs = s2_result[s2_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    rows: list[dict] = []
    use_s2_window = config.get("use_s2_window", False)

    # ── Serial fallback for small pair counts ────────────────────
    if n_pairs < 20:
        # Populate module globals so _process_one_pair_wfa works
        global _WORKER_PRICES, _WORKER_SIGNAL_FN
        _WORKER_PRICES = prices
        _WORKER_SIGNAL_FN = signal_fn
        for i, pair_row in enumerate(passed_pairs):
            result = _process_one_pair_wfa(
                pair_row,
                n_folds,
                fees,
                slope_window,
                use_s2_window,
                min_sharpe,
                min_win_rate,
                min_active_folds,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    wfa_expanding: {i + 1}/{n_pairs} " f"({n_pass} pass so far)",
                    flush=True,
                )
    else:
        # ── Parallel via fork + Pool ─────────────────────────────
        n_workers = min(os.cpu_count() or 4, 6)
        ctx = mp.get_context("fork")
        tasks = [
            (
                pr,
                n_folds,
                fees,
                slope_window,
                use_s2_window,
                min_sharpe,
                min_win_rate,
                min_active_folds,
            )
            for pr in passed_pairs
        ]
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(prices, signal_fn),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(
                    _process_one_pair_wfa_wrapper,
                    tasks,
                    chunksize=8,
                )
            ):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    wfa_expanding: {i + 1}/{n_pairs} "
                        f"({n_pass} pass so far)",
                        flush=True,
                    )

    empty_cols = [
        "pair",
        "numerator",
        "denominator",
        "mean_test_sharpe",
        "std_test_sharpe",
        "n_test_periods",
        "baseline_sharpe",
        "degradation",
        "passed",
        "val_method",
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_cols)
