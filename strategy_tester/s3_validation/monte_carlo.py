"""Monte Carlo trade-sequence resampling. Davey (2014)."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable

import numba
import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.vbt_runner import (
    backtest_numba_fold,
    build_is_oos_split,
    compute_ratio,
)
from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
)

SQRT_252 = np.float64(252.0) ** 0.5


@numba.njit(cache=True)
def _mc_resample_numba(
    oos_rets: np.ndarray,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated MC resampling: Sharpe + MaxDD per iteration."""
    n_oos = len(oos_rets)
    sharpes = np.empty(n_iter, dtype=np.float64)
    max_dds = np.empty(n_iter, dtype=np.float64)

    state = np.uint64(seed)
    for it in range(n_iter):
        # Resample with replacement
        s = np.float64(0.0)
        ss = np.float64(0.0)
        equity = np.float64(1.0)
        peak = np.float64(1.0)
        worst_dd = np.float64(0.0)

        for j in range(n_oos):
            state = state * np.uint64(6364136223846793005) + np.uint64(1)
            idx = int((state >> np.uint64(33)) % np.uint64(n_oos))
            r = oos_rets[idx]
            s += r
            ss += r * r

            # Running MaxDD
            equity *= (1.0 + r)
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak
            if dd < worst_dd:
                worst_dd = dd

        mean_r = s / n_oos
        var_r = (ss / n_oos - mean_r * mean_r) * n_oos / (n_oos - 1)
        std_r = var_r ** 0.5

        if std_r < 1e-10:
            sharpes[it] = 0.0
        else:
            sharpes[it] = (mean_r / std_r) * SQRT_252
        max_dds[it] = worst_dd

    return sharpes, max_dds


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


def _process_one_pair_monte_carlo(
    pair_row: dict,
    n_iter: int,
    fees: float,
    slope_window: int,
    is_ratio: float,
    random_state: int,
    min_sharpe: float = 0.0,
) -> dict | None:
    """Process a single pair through Monte Carlo resampling.

    Runs in a forked worker — reads prices / signal_fn from module globals.

    `n_oos_bars` semantics (METHODOLOGY_DECISIONS.md §1):
        The emitted `n_oos_bars` reflects the **post-cap** OOS length, NOT
        the full backtest. We cap at 630 bars (~2.5y) to prevent freezes on
        long-history assets; MC CI precision plateaus past that. Each MC
        resample draws from the same 630-bar pool, so the SR estimator IS
        based on 630 obs — passing 630 to PSR/DSR is methodologically
        correct. If you need the uncapped T for a different gate,
        recompute from the underlying return series, not from this row.
    """
    prices = _WORKER_PRICES
    signal_fn = _WORKER_SIGNAL_FN  # noqa: F841
    num = pair_row["numerator"]
    den = pair_row["denominator"]

    try:
        ratio, num_prices, common = compute_ratio(prices, pair_row)
    except KeyError:
        return None

    is_idx, oos_idx = build_is_oos_split(common, is_ratio)

    # Get OOS returns
    bt = backtest_numba_fold(
        num_prices.loc[oos_idx],
        ratio.loc[oos_idx],
        pair_row.get("window", 20),
        pair_row["entry_thresh"],
        pair_row["exit_thresh"],
        stop_pct=pair_row.get("stop_pct", 0.0),
        slope_min=pair_row.get("slope_min", 0.0),
        slope_window=slope_window,
        fees=fees,
        signal_fn=signal_fn,
    )
    oos_rets = bt["returns"]
    if isinstance(oos_rets, pd.Series):
        oos_rets = oos_rets.values
    oos_rets = np.asarray(oos_rets, dtype=np.float64)

    if len(oos_rets) < 20:
        return None

    # Cap OOS length to last 630 bars (~2.5yr) to prevent freeze on
    # long-history assets. MC CI precision plateaus with enough data.
    if len(oos_rets) > 630:
        oos_rets = oos_rets[-630:]

    n_oos_bars = int(len(oos_rets))  # post-cap T-days for PSR/DSR
    base_sharpe = annualized_sharpe(oos_rets)
    oos_cagr = geometric_cagr(oos_rets)
    oos_mdd = max_drawdown(oos_rets)

    # MC: Numba-accelerated resample + Sharpe + MaxDD
    mc_sharpes, mc_max_dds = _mc_resample_numba(
        oos_rets, n_iter, random_state,
    )

    ruin_prob = float(np.mean(mc_max_dds < -0.30))
    p05_dd = float(np.percentile(mc_max_dds, 5))
    mean_mc_sharpe = float(np.mean(mc_sharpes))

    return {
        "pair": pair_row["pair"],
        "numerator": num,
        "denominator": den,
        "mean_test_sharpe": round(mean_mc_sharpe, 4),
        "std_test_sharpe": round(
            float(np.std(mc_sharpes, ddof=1)), 4
        ),
        "n_test_periods": 1,  # single OOS test, not folds (iterations are internal)
        "n_oos_bars": n_oos_bars,  # post-630-cap T (see docstring)
        "baseline_sharpe": round(base_sharpe, 4),
        "degradation": round(
            1 - mean_mc_sharpe / base_sharpe
            if base_sharpe > 1e-6
            else float("nan"),
            4,
        ),
        "oos_cagr": round(oos_cagr, 4),
        "oos_max_dd": round(oos_mdd, 4),
        "passed": ruin_prob < 0.10 and base_sharpe >= min_sharpe,
        "val_method": "monte_carlo",
        "mc_ruin_prob": round(ruin_prob, 4),
        "mc_p05_max_dd": round(p05_dd, 4),
    }


def _process_one_pair_monte_carlo_wrapper(args: tuple) -> dict | None:
    """Unpack tuple for imap_unordered compatibility."""
    return _process_one_pair_monte_carlo(*args)


@register_stage("s3")
def monte_carlo(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """10K trade-sequence resampling. Ruin probability + DD distribution."""
    n_iter = config.get("mc_iterations", 10_000)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    is_ratio = config.get("is_ratio", 0.80)
    random_state = config.get("random_state", 42)
    min_sharpe = config.get("s3_min_oos_sharpe", 0.0)

    passed_pairs = s2_result[s2_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    rows: list[dict] = []

    tasks = [
        (pr, n_iter, fees, slope_window, is_ratio, random_state,
         min_sharpe)
        for pr in passed_pairs
    ]

    # ── Serial fallback for small pair counts ────────────────────
    if n_pairs < 20:
        _worker_init(prices, signal_fn)
        for i, args in enumerate(tasks):
            result = _process_one_pair_monte_carlo(*args)
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    monte_carlo: {i + 1}/{n_pairs} "
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
            maxtasksperchild=50,
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _process_one_pair_monte_carlo_wrapper, tasks, chunksize=1,
            )):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    monte_carlo: {i + 1}/{n_pairs} "
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
