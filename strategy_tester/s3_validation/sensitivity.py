"""Sensitivity analysis — ±20% parameter perturbation. Pardo §9.1-9.3."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.vbt_runner import (
    backtest_numba_fold,
    build_is_oos_split,
    compute_ratio,
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


def _process_one_pair_sensitivity(
    pair_row: dict,
    pct: float,
    fees: float,
    slope_window: int,
    is_ratio: float,
    min_sharpe: float = 0.0,
) -> dict | None:
    """Process a single pair through sensitivity perturbation.

    Runs in a forked worker — reads prices / signal_fn from module globals.

    Column semantics (METHODOLOGY_DECISIONS.md §1):
        `mean_test_sharpe` is the mean across N perturbed parameter sets,
        NOT a per-bar SR over time. The two count columns serve different
        downstream gates:
          - `n_test_periods` = perturbation count — the right ``n_obs`` if
            you treat `mean_test_sharpe` as a cross-perturbation Sharpe
            distribution (sensitivity's native semantics).
          - `n_oos_bars` = base backtest OOS bar count — for cross-
            validator consistency, so PSR/DSR can read the same column
            name. Using this overstates statistical strength relative to
            the cross-perturbation interpretation; only meaningful when
            treating `baseline_sharpe` as the headline SR over T bars.
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

    base_entry = pair_row["entry_thresh"]
    base_exit = pair_row["exit_thresh"]
    base_stop = pair_row.get("stop_pct", 0.0)
    base_slope = pair_row.get("slope_min", 0.0)
    window = pair_row.get("window", 20)

    # Baseline Sharpe on OOS
    base_bt = backtest_numba_fold(
        num_prices.loc[oos_idx],
        ratio.loc[oos_idx],
        window,
        base_entry,
        base_exit,
        stop_pct=base_stop,
        slope_min=base_slope,
        slope_window=slope_window,
        fees=fees,
        signal_fn=signal_fn,
    )
    base_sharpe = (
        base_bt["sharpe"] if not pd.isna(base_bt["sharpe"]) else 0.0
    )

    # OOS returns for CAGR / MaxDD.
    # NOTE on `n_oos_bars` (METHODOLOGY_DECISIONS.md §1):
    # We emit the base (unperturbed) OOS bar count for cross-validator
    # consistency. But `mean_test_sharpe` here is the mean across
    # perturbation runs (typically 4-8 SR estimates), NOT a per-bar SR
    # estimator. If a downstream gate runs PSR/DSR over this row, it will
    # interpret T as the base backtest length — which overstates the
    # statistical strength of the cross-perturbation mean. Use
    # `n_test_periods` (perturbation count) when treating this as a
    # cross-perturbation Sharpe distribution; use `n_oos_bars` only when
    # treating `baseline_sharpe` as the headline SR over T bars.
    base_rets = base_bt.get("returns")
    if base_rets is not None:
        if isinstance(base_rets, pd.Series):
            base_rets = base_rets.values
        base_rets = np.asarray(base_rets, dtype=np.float64)
        oos_cagr = geometric_cagr(base_rets)
        oos_mdd = max_drawdown(base_rets)
        n_oos_bars = int(len(base_rets))
    else:
        oos_cagr = 0.0
        oos_mdd = 0.0
        n_oos_bars = 0

    # Perturb each param by ±pct
    perturbed_sharpes = []
    params = [
        ("entry_thresh", base_entry),
        ("exit_thresh", base_exit),
        ("stop_pct", base_stop),
        ("slope_min", base_slope),
    ]
    for name, val in params:
        if val == 0.0:
            continue  # skip zero-valued params (not real for MA crossover)
        for direction in [-1, 1]:
            p_val = val * (1 + direction * pct)
            kw = {
                "entry_thresh": base_entry,
                "exit_thresh": base_exit,
                "stop_pct": base_stop,
                "slope_min": base_slope,
            }
            kw[name] = p_val
            bt = backtest_numba_fold(
                num_prices.loc[oos_idx],
                ratio.loc[oos_idx],
                window,
                kw["entry_thresh"],
                kw["exit_thresh"],
                stop_pct=max(0, kw["stop_pct"]),
                slope_min=kw["slope_min"],
                slope_window=slope_window,
                fees=fees,
                signal_fn=signal_fn,
            )
            s = bt["sharpe"] if not pd.isna(bt["sharpe"]) else 0.0
            perturbed_sharpes.append(s)

    if not perturbed_sharpes:
        # No perturbable params (MA crossover: all params = 0)
        # Pass based on base Sharpe only — no sensitivity to test
        mean_perturbed = base_sharpe
        std_perturbed = 0.0
        degradation = 0.0
        passed = base_sharpe >= min_sharpe
    else:
        mean_perturbed = float(np.mean(perturbed_sharpes))
        std_perturbed = (
            float(np.std(perturbed_sharpes, ddof=1))
            if len(perturbed_sharpes) > 1
            else 0.0
        )
        degradation = (
            1 - mean_perturbed / base_sharpe
            if base_sharpe > 1e-6
            else float("nan")
        )
        passed = degradation < 0.50 and base_sharpe >= min_sharpe

    return {
        "pair": pair_row["pair"],
        "numerator": num,
        "denominator": den,
        "mean_test_sharpe": round(mean_perturbed, 4),
        "std_test_sharpe": round(std_perturbed, 4),
        "n_test_periods": len(perturbed_sharpes),  # perturbation count
        "n_oos_bars": n_oos_bars,                  # base backtest T (see docstring)
        "baseline_sharpe": round(base_sharpe, 4),
        "degradation": round(degradation, 4),
        "oos_cagr": round(oos_cagr, 4),
        "oos_max_dd": round(oos_mdd, 4),
        "passed": passed,
        "val_method": "sensitivity",
    }


def _process_one_pair_sensitivity_wrapper(args: tuple) -> dict | None:
    """Unpack tuple for imap_unordered compatibility."""
    return _process_one_pair_sensitivity(*args)


@register_stage("s3")
def sensitivity(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """±sensitivity_pct perturbation on each param. Plateau check."""
    pct = config.get("sensitivity_pct", 0.20)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    is_ratio = config.get("is_ratio", 0.80)
    min_sharpe = config.get("s3_min_oos_sharpe", 0.0)

    passed_pairs = s2_result[s2_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    rows: list[dict] = []

    tasks = [
        (pr, pct, fees, slope_window, is_ratio, min_sharpe)
        for pr in passed_pairs
    ]

    # ── Serial fallback for small pair counts ────────────────────
    if n_pairs < 20:
        _worker_init(prices, signal_fn)
        for i, args in enumerate(tasks):
            result = _process_one_pair_sensitivity(*args)
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    sensitivity: {i + 1}/{n_pairs} "
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
                _process_one_pair_sensitivity_wrapper, tasks, chunksize=8,
            )):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    sensitivity: {i + 1}/{n_pairs} "
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
