"""CPCV validation. LdP AFML Ch.12.

Combinatorial Purged Cross-Validation: 10 folds, C(10,2) = 45 combos,
stitched into C(9,1) = 9 complete backtest paths via 1-factorization.
Each path covers all 10 folds exactly once. Sharpe computed per path,
then aggregated across the 9 independent paths.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from itertools import combinations
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
    geometric_cagr,
    max_drawdown,
)

# ---------------------------------------------------------------------------
# Worker initialiser — fork context inherits parent memory, but we store
# references via globals for clarity (grid_ma.py pattern).
# ---------------------------------------------------------------------------
_WORKER_PRICES: pd.DataFrame | None = None
_WORKER_SIGNAL_FN: Callable | None = None


def _worker_init(
    prices: pd.DataFrame,
    signal_fn: Callable | None,
) -> None:
    global _WORKER_PRICES, _WORKER_SIGNAL_FN
    _WORKER_PRICES = prices
    _WORKER_SIGNAL_FN = signal_fn


def _build_1_factorization(n_folds: int) -> list[list[tuple[int, int]]]:
    """Build n_folds-1 perfect matchings of K_{n_folds} (round-robin).

    Each matching = 5 combos covering all 10 folds exactly once.
    Together the 9 matchings partition all C(10,2) = 45 edges.

    Returns list of 9 matchings, each a list of 5 (fold_i, fold_j) pairs.
    """
    pivot = n_folds - 1
    others = list(range(n_folds - 1))  # [0..8]
    paths: list[list[tuple[int, int]]] = []

    for r in range(n_folds - 1):
        matching: list[tuple[int, int]] = []
        rotated = others[r:] + others[:r]
        # Pair pivot with first rotated vertex
        matching.append(tuple(sorted((pivot, rotated[0]))))
        # Pair remaining symmetrically
        half = len(rotated) // 2
        for k in range(1, half + 1):
            a, b = rotated[k], rotated[-(k)]
            matching.append(tuple(sorted((a, b))))
        paths.append(matching)

    return paths


def build_cpcv_folds(
    common_idx: pd.Index,
    n_folds: int = 10,
    n_test_folds: int = 2,
    purge_bars: int = 20,
    embargo_bars: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build C(n_folds, n_test_folds) CPCV combinations with purge+embargo.

    Kept for backward compatibility / external callers.
    """
    n = len(common_idx)
    chunk = n // n_folds
    if chunk < 20:
        return []

    fold_starts = [i * chunk for i in range(n_folds)]
    fold_ends = fold_starts[1:] + [n]

    all_combos = list(combinations(range(n_folds), n_test_folds))
    result = []

    for test_indices in all_combos:
        test_set = set(test_indices)
        raw_test: set[int] = set()
        for fi in test_indices:
            s, e = fold_starts[fi], fold_ends[fi]
            raw_test.update(range(s, e))

        purge: set[int] = set()
        for fi in test_indices:
            s, e = fold_starts[fi], fold_ends[fi]
            for p in range(max(0, s - purge_bars), min(n, s + purge_bars)):
                purge.add(p)
            for p in range(max(0, e - purge_bars), min(n, e + purge_bars)):
                purge.add(p)

        embargo: set[int] = set()
        for fi in test_indices:
            e = fold_ends[fi]
            for p in range(e, min(n, e + embargo_bars)):
                embargo.add(p)

        test_positions = sorted(raw_test)
        excluded = raw_test | purge | embargo

        train_positions = []
        for fi in range(n_folds):
            if fi in test_set:
                continue
            s, e = fold_starts[fi], fold_ends[fi]
            for p in range(s, e):
                if p not in excluded:
                    train_positions.append(p)

        min_test = max(20, chunk // 2)
        if len(test_positions) < min_test or len(train_positions) < 100:
            continue

        result.append((
            np.array(train_positions, dtype=np.intp),
            np.array(test_positions, dtype=np.intp),
        ))

    return result


# ---------------------------------------------------------------------------
# Per-pair worker function
# ---------------------------------------------------------------------------

def _process_one_pair_cpcv(
    pair_row: dict,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
    fees: float,
    slope_window: int,
    config_dict: dict,
) -> dict | None:
    """Process a single pair through proper CPCV. Runs in worker process.

    When use_s2_window=True: 10 OOS backtests (all 9 paths identical).
    When use_s2_window=False: 90 OOS backtests (45 combos × 2 test folds),
    stitched into 9 independent paths via 1-factorization.
    """
    prices = _WORKER_PRICES
    signal_fn = _WORKER_SIGNAL_FN
    use_s2_window = config_dict.get("use_s2_window", False)
    min_sharpe = config_dict.get("s3_min_oos_sharpe", 0.0)

    try:
        ratio, num_prices, common = compute_ratio(prices, pair_row)
    except KeyError:
        return None

    n = len(common)
    chunk = n // n_folds
    if chunk < 20:
        return None
    fold_starts = [k * chunk for k in range(n_folds)]
    fold_ends = fold_starts[1:] + [n]

    def fold_slice(fi: int) -> pd.Index:
        return common[fold_starts[fi]:fold_ends[fi]]

    def build_train_mask(excluded_folds: tuple[int, ...]) -> np.ndarray:
        mask = np.ones(n, dtype=bool)
        for fi in excluded_folds:
            mask[fold_starts[fi]:fold_ends[fi]] = False
            for p in range(
                max(0, fold_starts[fi] - purge_bars),
                min(n, fold_starts[fi] + purge_bars),
            ):
                mask[p] = False
            for p in range(
                max(0, fold_ends[fi] - purge_bars),
                min(n, fold_ends[fi] + purge_bars),
            ):
                mask[p] = False
            for p in range(
                fold_ends[fi],
                min(n, fold_ends[fi] + embargo_bars),
            ):
                mask[p] = False
        return mask

    def get_window(train_idx: pd.Index) -> int:
        if use_s2_window:
            return int(pair_row.get("window", 20))
        hl = compute_halflife(ratio.loc[train_idx])
        return (
            min(max(int(hl * 0.5), 10), 252)
            if not np.isnan(hl)
            else int(pair_row.get("window", 20))
        )

    def run_oos(test_idx: pd.Index, w: int) -> np.ndarray | None:
        bt = backtest_numba_fold(
            num_prices.loc[test_idx], ratio.loc[test_idx],
            w, pair_row["entry_thresh"], pair_row["exit_thresh"],
            stop_pct=pair_row.get("stop_pct", 0.0),
            slope_min=pair_row.get("slope_min", 0.0),
            slope_window=slope_window, fees=fees, signal_fn=signal_fn,
        )
        rets = bt.get("returns")
        if rets is None:
            return None
        if isinstance(rets, pd.Series):
            rets = rets.values
        return np.asarray(rets, dtype=np.float64)

    # ------------------------------------------------------------------
    # CPCV: 45 combos → 9 stitched paths
    # ------------------------------------------------------------------
    all_combos = list(combinations(range(n_folds), 2))

    if use_s2_window:
        # SHORTCUT: params fixed from S2 — all combos give same per-fold
        # OOS. Run 10 fold backtests, all 9 paths are identical.
        w = int(pair_row.get("window", 20))
        fold_returns: dict[int, np.ndarray] = {}
        for fi in range(n_folds):
            fidx = fold_slice(fi)
            if len(fidx) < 20:
                continue
            rets = run_oos(fidx, w)
            if rets is not None and len(rets) > 0:
                fold_returns[fi] = rets

        if len(fold_returns) < 3:
            return None

        # Stitch chronologically — single path (all 9 identical)
        stitched = np.concatenate(
            [fold_returns[fi] for fi in sorted(fold_returns)]
        )
        stitched_sr = annualized_sharpe(stitched)
        stitched_cagr = geometric_cagr(stitched)
        stitched_max_dd = max_drawdown(stitched)
        n_oos_bars = int(len(stitched))
        per_fold_sr = [
            annualized_sharpe(fold_returns[fi])
            for fi in sorted(fold_returns) if len(fold_returns[fi]) > 10
        ]
        # 9 identical paths → same Sharpe
        path_sharpes = [stitched_sr] * min(n_folds - 1, 9)

    else:
        # FULL CPCV: run per-combo, per-fold OOS backtests.
        # combo_fold_returns[(combo_tuple, fold_idx)] = returns
        combo_fold_returns: dict[
            tuple[tuple[int, int], int], np.ndarray
        ] = {}

        for combo in all_combos:
            fi, fj = combo
            train_mask = build_train_mask(combo)
            train_idx = common[train_mask]
            if len(train_idx) < 100:
                continue
            w = get_window(train_idx)

            for test_fi in combo:
                fidx = fold_slice(test_fi)
                if len(fidx) < 20:
                    continue
                rets = run_oos(fidx, w)
                if rets is not None and len(rets) > 0:
                    combo_fold_returns[(combo, test_fi)] = rets

        # Build 9 paths via 1-factorization of K_10
        matchings = _build_1_factorization(n_folds)
        path_sharpes = []
        all_path_rets: list[np.ndarray] = []
        for matching in matchings:
            path_rets: list[tuple[int, np.ndarray]] = []
            for combo in matching:
                for fi in sorted(combo):
                    key = (combo, fi)
                    if key in combo_fold_returns:
                        path_rets.append((fi, combo_fold_returns[key]))
            if len(path_rets) < 3:
                continue
            # Sort by fold index → chronological order
            path_rets.sort(key=lambda x: x[0])
            stitched = np.concatenate([r for _, r in path_rets])
            path_sharpes.append(annualized_sharpe(stitched))
            all_path_rets.append(stitched)

        if not path_sharpes:
            return None

        # Aggregate OOS returns from first complete path
        stitched_cagr = geometric_cagr(all_path_rets[0])
        stitched_max_dd = max_drawdown(all_path_rets[0])
        n_oos_bars = int(len(all_path_rets[0]))

        per_fold_sr = path_sharpes  # each path is a complete estimate

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    mean_oos = float(np.mean(path_sharpes))
    # Use per_fold_sr for std (path_sharpes can be identical when
    # use_s2_window=True, but per-fold Sharpes always differ)
    std_oos = (
        float(np.std(per_fold_sr, ddof=1))
        if len(per_fold_sr) > 1
        else 0.0
    )
    # IS Sharpe from S2 (avoids re-running IS backtests per combo)
    mean_is = float(pair_row.get("is_sharpe", 0.0))
    if np.isinf(mean_oos) or np.isnan(mean_oos):
        mean_oos = 0.0
    if np.isinf(mean_is) or np.isnan(mean_is):
        mean_is = 0.0

    pct_positive = (
        sum(1 for s in per_fold_sr if s > 0) / len(per_fold_sr)
        if per_fold_sr
        else 0.0
    )
    degradation = (
        round(1 - mean_oos / mean_is, 4)
        if mean_is > 1e-6
        else float("nan")
    )

    return {
        "pair": pair_row["pair"],
        "numerator": pair_row["numerator"],
        "denominator": pair_row["denominator"],
        "mean_test_sharpe": round(mean_oos, 4),
        "std_test_sharpe": round(std_oos, 4),
        "n_test_periods": len(per_fold_sr),  # fold count, not path count
        "n_oos_bars": n_oos_bars,
        "baseline_sharpe": round(mean_is, 4),
        "degradation": degradation,
        "oos_cagr": round(stitched_cagr, 4),
        "oos_max_dd": round(stitched_max_dd, 4),
        "passed": mean_oos >= min_sharpe and pct_positive >= 0.5,
        "val_method": "cpcv",
        "pct_paths_positive": round(pct_positive, 4),
    }


def _process_one_pair_cpcv_wrapper(args: tuple) -> dict | None:
    """Top-level wrapper for imap_unordered (must be picklable)."""
    return _process_one_pair_cpcv(*args)


@register_stage("s3")
def cpcv(
    prices: pd.DataFrame,
    s2_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """10-fold CPCV, 9 stitched paths. LdP AFML Ch.12."""
    n_folds = config.get("cpcv_n_folds", 10)
    purge = config.get("cpcv_purge", 20)
    embargo = config.get("cpcv_embargo", 5)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)

    passed_pairs = s2_result[s2_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    rows: list[dict] = []

    # Extract only picklable scalars from config (no lambdas/callables)
    config_dict = {
        k: v for k, v in config.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }

    tasks = [
        (pr, n_folds, purge, embargo, fees, slope_window, config_dict)
        for pr in passed_pairs
    ]

    # Serial fallback for small pair counts
    if n_pairs < 20:
        _worker_init(prices, signal_fn)
        for i, args in enumerate(tasks):
            result = _process_one_pair_cpcv(*args)
            if result is not None:
                rows.append(result)
            if (i + 1) % 10 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                print(
                    f"    cpcv: {i + 1}/{n_pairs} "
                    f"({n_pass} pass so far)",
                    flush=True,
                )
    else:
        # Parallel: fork context inherits parent memory
        n_workers = min(os.cpu_count() or 4, 6)
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(prices, signal_fn),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _process_one_pair_cpcv_wrapper,
                tasks,
                chunksize=8,
            )):
                if result is not None:
                    rows.append(result)
                if (i + 1) % 50 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    print(
                        f"    cpcv: {i + 1}/{n_pairs} "
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
