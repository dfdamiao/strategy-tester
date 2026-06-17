"""Bootstrap CI on Sharpe. Carver (2015)."""
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
def _bootstrap_sharpes_numba(
    blocks: np.ndarray,  # (n_blocks, block_size)
    n_iter: int,
    n_blocks: int,
    seed: int,
) -> np.ndarray:
    """Numba-accelerated block bootstrap Sharpe computation."""
    block_size = blocks.shape[1]
    sample_len = n_blocks * block_size
    sharpes = np.empty(n_iter, dtype=np.float64)

    # Simple LCG random for Numba (no rng.integers in njit)
    state = np.uint64(seed)
    for it in range(n_iter):
        # Build resampled return array
        sample = np.empty(sample_len, dtype=np.float64)
        pos = 0
        for b in range(n_blocks):
            # LCG step
            state = state * np.uint64(6364136223846793005) + np.uint64(1)
            idx = int((state >> np.uint64(33)) % np.uint64(n_blocks))
            for k in range(block_size):
                sample[pos] = blocks[idx, k]
                pos += 1

        # Sharpe: mean / std * sqrt(252)
        s = np.float64(0.0)
        for j in range(sample_len):
            s += sample[j]
        mean_r = s / sample_len

        ss = np.float64(0.0)
        for j in range(sample_len):
            ss += (sample[j] - mean_r) ** 2
        var_r = ss / (sample_len - 1)
        std_r = var_r ** 0.5

        if std_r < 1e-10:
            sharpes[it] = 0.0
        else:
            sharpes[it] = (mean_r / std_r) * SQRT_252

    return sharpes


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


def _process_one_pair_bootstrap(
    pair_row: dict,
    n_iter: int,
    fees: float,
    slope_window: int,
    is_ratio: float,
    random_state: int,
    min_sharpe: float = 0.0,
) -> dict | None:
    """Process a single pair through bootstrap CI.

    Runs in a forked worker — reads prices / signal_fn from module globals.
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

    n_oos_bars = int(len(oos_rets))  # T-days for PSR/DSR
    base_sharpe = annualized_sharpe(oos_rets)
    oos_cagr = geometric_cagr(oos_rets)
    oos_mdd = max_drawdown(oos_rets)

    # Bootstrap monthly blocks — cap at 30 blocks (Politis & Romano 1994:
    # CI precision plateaus beyond ~20-30 resampling units)
    block_size = 21  # ~1 month
    n_blocks = min(len(oos_rets) // block_size, 30)
    if n_blocks < 3:
        return None

    # Stack blocks into a 2D array
    blocks_arr = np.array([
        oos_rets[j * block_size: (j + 1) * block_size]
        for j in range(n_blocks)
    ])  # shape: (n_blocks, block_size)

    boot_sharpes = _bootstrap_sharpes_numba(
        blocks_arr, n_iter, n_blocks, random_state,
    )
    ci_lower = float(np.percentile(boot_sharpes, 2.5))
    ci_upper = float(np.percentile(boot_sharpes, 97.5))
    mean_boot = float(np.mean(boot_sharpes))

    return {
        "pair": pair_row["pair"],
        "numerator": num,
        "denominator": den,
        "mean_test_sharpe": round(mean_boot, 4),
        "std_test_sharpe": round(
            float(np.std(boot_sharpes, ddof=1)), 4
        ),
        "n_test_periods": 1,  # single OOS test, not folds (iterations are internal)
        "n_oos_bars": n_oos_bars,
        "baseline_sharpe": round(base_sharpe, 4),
        "degradation": round(
            1 - mean_boot / base_sharpe
            if base_sharpe > 1e-6
            else float("nan"),
            4,
        ),
        "oos_cagr": round(oos_cagr, 4),
        "oos_max_dd": round(oos_mdd, 4),
        "passed": ci_lower > 0 and base_sharpe >= min_sharpe,
        "val_method": "bootstrap_ci",
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def _process_one_pair_bootstrap_wrapper(args: tuple) -> dict | None:
    """Unpack tuple for imap_unordered compatibility."""
    return _process_one_pair_bootstrap(*args)


@register_stage("s3")
def bootstrap_ci(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """10K monthly return resamples. 95% CI on Sharpe."""
    n_iter = config.get("bootstrap_iterations", 10_000)
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
            result = _process_one_pair_bootstrap(*args)
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    bootstrap_ci: {i + 1}/{n_pairs} "
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
            maxtasksperchild=50,  # recycle workers to avoid memory bloat
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _process_one_pair_bootstrap_wrapper, tasks, chunksize=1,
            )):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    bootstrap_ci: {i + 1}/{n_pairs} "
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
