"""Simple 80/20 IS/OOS split validation. Chan Ch.2-3."""
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
    build_is_oos_split,
)
from strategy_tester.backtest.metrics import geometric_cagr, max_drawdown

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


def _process_one_pair_chan_is_oos(
    pair_row: dict,
    is_ratio: float,
    fees: float,
    slope_window: int,
    use_s2_window: bool,
    min_sharpe: float = 0.0,
) -> dict | None:
    """Process a single pair through 80/20 IS/OOS split.

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

    is_idx, oos_idx = build_is_oos_split(common, is_ratio)

    # Re-estimate halflife on IS
    if use_s2_window:
        window = int(pair_row.get("window", 20))
    else:
        hl = compute_halflife(ratio.loc[is_idx])
        window = (
            min(max(int(hl * 0.5), 10), 252)
            if not np.isnan(hl)
            else pair_row.get("window", 20)
        )

    # Backtest on IS and OOS
    is_bt = backtest_numba_fold(
        num_prices.loc[is_idx],
        ratio.loc[is_idx],
        window,
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
        window,
        pair_row["entry_thresh"],
        pair_row["exit_thresh"],
        stop_pct=pair_row.get("stop_pct", 0.0),
        slope_min=pair_row.get("slope_min", 0.0),
        slope_window=slope_window,
        fees=fees,
        signal_fn=signal_fn,
    )

    # OOS returns for CAGR / MaxDD
    oos_rets = oos_bt.get("returns")
    if oos_rets is not None:
        if isinstance(oos_rets, pd.Series):
            oos_rets = oos_rets.values
        oos_rets = np.asarray(oos_rets, dtype=np.float64)
        oos_cagr = geometric_cagr(oos_rets)
        oos_mdd = max_drawdown(oos_rets)
        n_oos_bars = int(len(oos_rets))
    else:
        oos_cagr = 0.0
        oos_mdd = 0.0
        n_oos_bars = 0

    is_sharpe = (
        is_bt["sharpe"] if not pd.isna(is_bt["sharpe"]) else 0.0
    )
    oos_sharpe = (
        oos_bt["sharpe"] if not pd.isna(oos_bt["sharpe"]) else 0.0
    )
    degradation = (
        1 - oos_sharpe / is_sharpe
        if is_sharpe > 1e-6
        else float("nan")
    )

    # Chan AT (2013) Ch.2-3: SR > 0.5 "worth pursuing" (informational, not gated)
    chan_worth_pursuing = oos_sharpe >= 0.5

    return {
        "pair": pair_row["pair"],
        "numerator": num,
        "denominator": den,
        "mean_test_sharpe": round(oos_sharpe, 4),
        "std_test_sharpe": 0.0,
        "n_test_periods": 1,
        "n_oos_bars": n_oos_bars,
        "baseline_sharpe": round(is_sharpe, 4),
        "degradation": round(degradation, 4),
        "oos_cagr": round(oos_cagr, 4),
        "oos_max_dd": round(oos_mdd, 4),
        "chan_worth_pursuing": chan_worth_pursuing,
        "passed": oos_sharpe >= min_sharpe,
        "val_method": "chan_is_oos",
    }


def _process_one_pair_chan_is_oos_wrapper(args: tuple) -> dict | None:
    """Unpack tuple for imap_unordered compatibility."""
    return _process_one_pair_chan_is_oos(*args)


@register_stage("s3")
def chan_is_oos(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """Simple 80/20 IS/OOS validation. Re-estimates halflife on IS."""
    is_ratio = config.get("is_ratio", 0.80)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    min_sharpe = config.get("s3_min_oos_sharpe", 0.0)

    passed_pairs = s2_result[s2_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    rows: list[dict] = []
    use_s2_window = config.get("use_s2_window", False)

    tasks = [
        (pr, is_ratio, fees, slope_window, use_s2_window, min_sharpe)
        for pr in passed_pairs
    ]

    # ── Serial fallback for small pair counts ────────────────────
    if n_pairs < 20:
        _worker_init(prices, signal_fn)
        for i, args in enumerate(tasks):
            result = _process_one_pair_chan_is_oos(*args)
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    chan_is_oos: {i + 1}/{n_pairs} "
                    f"({n_pass} pass so far)",
                    flush=True,
                )
    else:
        # ── Parallel via fork + Pool ─────────────────────────────
        n_workers = min(os.cpu_count() or 4, 6)
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(prices, signal_fn),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _process_one_pair_chan_is_oos_wrapper, tasks, chunksize=8,
            )):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    chan_is_oos: {i + 1}/{n_pairs} "
                        f"({n_pass} pass so far)",
                        flush=True,
                    )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "pair", "numerator", "denominator",
            "mean_test_sharpe", "std_test_sharpe",
            "n_test_periods", "baseline_sharpe",
            "degradation", "passed", "val_method",
        ]
    )
