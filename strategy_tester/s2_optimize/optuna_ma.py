"""Optuna TPE optimizer for MA crossover — single parameter (window).

Optimizes MA window in [ma_low, ma_high] via Bayesian search.
For each trial, computes MA signal and backtests on IS data.
Best window evaluated on OOS data.

Handles both singles (MA on price) and pairs (MA on ratio).

References:
    Chan, Quantitative Trading (2008) — per-asset window optimization
    Murphy, Technical Analysis (1999) Ch.9 — MA crossover
    LdP, AFML (2018) Ch.14 — penalized Sharpe
"""
from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import (
    backtest_vbt_precomputed,
    build_is_oos_split,
    calculate_penalized_sharpe,
    compute_halflife,
)
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _get_signal_module(signal_fn: Callable | None):
    """Get the module of a signal function for precompute/apply."""
    if signal_fn is None:
        return None
    return importlib.import_module(signal_fn.__module__)


def _build_series(
    prices: pd.DataFrame, pair_row: dict,
) -> tuple[pd.Series, pd.Series, pd.Index] | None:
    """Build signal base + numerator close for singles or pairs.

    Returns (signal_base, num_close, common_idx) or None.
    signal_base: price for singles, ratio for pairs.
    num_close: numerator close (what we trade).
    """
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    asset_type = pair_row.get("asset_type", "pair")

    if num not in prices.columns:
        return None

    p_num = prices[num].dropna()

    if asset_type == "single" or num == den:
        # Single: MA on price directly
        return p_num, p_num, p_num.index

    if den not in prices.columns:
        return None

    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < 252:
        return None
    ratio = p_num.loc[common] / p_den.loc[common]
    return ratio, p_num.loc[common], common


def _process_one_pair(
    prices: pd.DataFrame,
    pair_row: dict,
    ma_low: int,
    ma_high: int,
    is_ratio: float,
    fees: float,
    slope_window: int,
    n_trials: int,
    random_state: int,
    min_is_trades: int,
    min_oos_trades: int,
    signal_fn: Callable | None,
    signal_module,
) -> dict | None:
    """Optuna TPE optimization of MA window for one pair/single."""
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pair_name = pair_row["pair"]
    num = pair_row["numerator"]
    den = pair_row["denominator"]

    series_data = _build_series(prices, pair_row)
    if series_data is None:
        return None
    signal_base, num_close, common = series_data

    is_idx, oos_idx = build_is_oos_split(common, is_ratio)
    base_is = signal_base.loc[is_idx]
    num_is = num_close.loc[is_idx]
    base_oos = signal_base.loc[oos_idx]
    num_oos = num_close.loc[oos_idx]

    if len(is_idx) < ma_high + 10 or len(oos_idx) < 60:
        return None

    # Resolve signal module (ma_crossover expected)
    has_precompute = (
        signal_module is not None
        and hasattr(signal_module, "precompute")
        and hasattr(signal_module, "apply_thresholds")
    )

    # n_params = 1 (window only) for penalized Sharpe
    n_params = 1

    def objective(trial: "optuna.Trial") -> float:
        w = trial.suggest_int("window", ma_low, ma_high)

        if has_precompute:
            pre = signal_module.precompute(base_is, w, slope_window)
            entries, exits = signal_module.apply_thresholds(
                pre, 0.0, 0.0, 0.0,
            )
            bt = backtest_vbt_precomputed(
                num_is, entries, exits, stop_pct=0.0, fees=fees,
            )
        else:
            # Fallback: compute MA signal inline
            ma = base_is.rolling(w, min_periods=w).mean()
            signal = (base_is > ma).astype(int)
            entries = (signal == 1).shift(1, fill_value=False)
            exits = (signal == 0).shift(1, fill_value=False)
            bt = backtest_vbt_precomputed(
                num_is, entries, exits, stop_pct=0.0, fees=fees,
            )

        if pd.isna(bt["sharpe"]) or bt["n_trades"] < min_is_trades:
            return -999.0
        return calculate_penalized_sharpe(
            bt["sharpe"], n_params, bt["n_trades"],
        )

    sampler = TPESampler(seed=random_state, multivariate=True)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
    )
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False,
    )

    if study.best_value <= -999:
        return None

    best_window = study.best_params["window"]

    # Evaluate best window on IS and OOS
    if has_precompute:
        pre_is = signal_module.precompute(
            base_is, best_window, slope_window,
        )
        is_e, is_x = signal_module.apply_thresholds(
            pre_is, 0.0, 0.0, 0.0,
        )
        best_is = backtest_vbt_precomputed(
            num_is, is_e, is_x, stop_pct=0.0, fees=fees,
        )

        # Warmup: prepend IS-tail bars to prime the MA
        warmup = best_window
        warmup_base = signal_base.loc[is_idx].iloc[-warmup:]
        ext_base = pd.concat([warmup_base, base_oos])
        pre_oos = signal_module.precompute(ext_base, best_window, slope_window)
        # Trim warmup from signal
        oos_offset = len(warmup_base)
        full_e, full_x = signal_module.apply_thresholds(pre_oos, 0.0, 0.0, 0.0)
        oos_e = full_e.iloc[oos_offset:]
        oos_x = full_x.iloc[oos_offset:]
        oos_bt = backtest_vbt_precomputed(
            num_oos, oos_e, oos_x, stop_pct=0.0, fees=fees,
        )
    else:
        # Inline fallback — IS
        ma_is = base_is.rolling(
            best_window, min_periods=best_window,
        ).mean()
        sig_is = (base_is > ma_is).astype(int)
        best_is = backtest_vbt_precomputed(
            num_is,
            (sig_is == 1).shift(1, fill_value=False),
            (sig_is == 0).shift(1, fill_value=False),
            stop_pct=0.0, fees=fees,
        )
        # Inline fallback — OOS (warmup: prepend IS-tail bars)
        warmup_base = signal_base.loc[is_idx].iloc[-best_window:]
        ext_base = pd.concat([warmup_base, base_oos])
        ma_oos = ext_base.rolling(best_window, min_periods=best_window).mean()
        sig_oos = (ext_base > ma_oos).astype(int)
        # Trim warmup
        oos_offset = len(warmup_base)
        sig_oos = sig_oos.iloc[oos_offset:]
        oos_bt = backtest_vbt_precomputed(
            num_oos,
            (sig_oos == 1).shift(1, fill_value=False),
            (sig_oos == 0).shift(1, fill_value=False),
            stop_pct=0.0, fees=fees,
        )

    # Halflife computed for interface compliance (informational)
    hl = compute_halflife(signal_base)
    if np.isnan(hl):
        hl = float("nan")

    passed = (
        not pd.isna(oos_bt["sharpe"])
        and oos_bt["sharpe"] > 0
        and oos_bt["n_trades"] >= min_oos_trades
    )

    return {
        "pair": pair_name,
        "numerator": num,
        "denominator": den,
        "halflife": round(hl, 2) if not np.isnan(hl) else float("nan"),
        "window": best_window,
        "entry_thresh": 0.0,
        "exit_thresh": 0.0,
        "stop_pct": 0.0,
        "slope_min": 0.0,
        "is_sharpe": best_is["sharpe"],
        "is_penalized_sharpe": round(study.best_value, 4),
        "is_trades": best_is["n_trades"],
        "oos_sharpe": oos_bt["sharpe"],
        "oos_trades": oos_bt["n_trades"],
        "passed": passed,
        "signal_method": (
            signal_fn.__name__ if signal_fn else "ma_crossover"
        ),
        "optim_method": "optuna_ma",
    }


@register_stage("s2_optimize")
def optuna_ma(
    prices: pd.DataFrame,
    s1_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """Optuna TPE optimizer for MA window [ma_low, ma_high].

    Single parameter search — optimizes only the MA lookback window.
    Uses ma_crossover signal method (or signal_fn if provided).

    Config keys:
        ma_low (10): minimum MA window
        ma_high (270): maximum MA window
        optuna_n_trials (50): Bayesian search budget
        is_ratio (0.80): IS/OOS split ratio
        cost_per_side (0.001): transaction costs (10 bps)
        min_is_trades (3): minimum IS trades for valid trial
        min_oos_trades (3): minimum OOS trades to pass

    References:
        Chan, Quantitative Trading (2008) — per-asset optimization
        LdP, AFML (2018) Ch.14 — penalized Sharpe, trial budget
    """
    ma_low = config.get("ma_low", 10)
    ma_high = config.get("ma_high", 270)
    is_ratio = config.get("is_ratio", 0.80)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    n_trials = config.get("optuna_n_trials", 50)
    random_state = config.get("random_state", 42)
    min_is_trades = config.get("min_is_trades", 3)
    min_oos_trades = config.get("min_oos_trades", 3)
    parallel = config.get("parallel", True)

    passed_pairs = s1_result[s1_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)

    signal_module = _get_signal_module(signal_fn)
    _log(f"optuna_ma: {n_pairs} pairs x {n_trials} trials "
         f"(window [{ma_low}, {ma_high}])")

    rows: list[dict] = []

    if parallel and n_pairs > 10:
        n_workers = min(os.cpu_count() or 4, 6)
        _log(f"optuna_ma: {n_workers} workers started")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_pair, prices, pr,
                    ma_low, ma_high, is_ratio, fees, slope_window,
                    n_trials, random_state,
                    min_is_trades, min_oos_trades,
                    signal_fn, signal_module,
                ): pr
                for pr in passed_pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 10 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    _log(
                        f"optuna_ma: {i + 1}/{n_pairs} "
                        f"({n_pass} pass so far)"
                    )
    else:
        for i, pr in enumerate(passed_pairs):
            result = _process_one_pair(
                prices, pr,
                ma_low, ma_high, is_ratio, fees, slope_window,
                n_trials, random_state,
                min_is_trades, min_oos_trades,
                signal_fn, signal_module,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 50 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                _log(
                    f"optuna_ma: {i + 1}/{n_pairs} "
                    f"({n_pass} pass so far)"
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "pair", "numerator", "denominator", "halflife",
            "window", "entry_thresh", "exit_thresh", "stop_pct",
            "slope_min", "is_sharpe", "is_penalized_sharpe",
            "is_trades", "oos_sharpe", "oos_trades", "passed",
            "signal_method", "optim_method",
        ]
    )
