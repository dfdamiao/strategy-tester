"""Canonical empirical rho_bar + sr_var_full_grid helper for per_asset DSR.

Per docs/rules.md §3g, per_asset DSR computation MUST use:
  - n_trials = full grid size (not qualifying-count)
  - sr_variance_annualized = Var[IS Sharpe across all non-degenerate combos]
  - rho_bar = empirical mean pairwise correlation of trial return series

This module provides the reference implementation. Strategy `sweep_s2.py`
scripts should call `compute_dsr_inputs(...)` after S2 sweep completes and
write the result to `{results_dir}/{mode}_dsr_inputs.csv` so the S4 stage
can load the per-unit (rho_bar, sr_var_full_grid) values.

Canonical schema for `{mode}_dsr_inputs.csv`:
    unit                   str   "TICKER" or "NUM/DEN"
    cohort                 str   "singles" or "ratios"
    rho_bar                float Mean pairwise correlation in [0, 1)
    n_valid_combos         int   Count of non-zero-variance combos (out of grid_size)
    sr_var_full_grid       float Var[IS Sharpe] across all valid combos
    is_sharpe_max_full     float Max IS Sharpe across all valid combos (informational)
    grid_size              int   Full search-space size (e.g., 96)

Usage example
-------------
    from strategy_tester.s4_significance.compute_rho_bar import (
        compute_dsr_inputs,
    )

    def my_signal_pos_raw(price, **params) -> np.ndarray:
        ...  # strategy-specific signal kernel returning {0, 1}

    df = compute_dsr_inputs(
        universe=universe_df,                # rows with unit + cohort + leg tickers
        close=close_prices_df,               # MDA / yfinance output
        combos=list_of_param_dicts,          # full grid (e.g., 96 combos)
        compute_pos_raw=my_signal_pos_raw,   # callable(price, **params) -> np.ndarray
        build_unit_price=cm.build_unit_price,
        costs_bps=10.0,
        is_ratio=0.8,
        annualization=252,
        n_workers=8,
    )
    df.to_csv(results_dir / "per_asset_dsr_inputs.csv", index=False)

Then in the S4 loop:
    dsr_inputs = pd.read_csv(results_dir / "per_asset_dsr_inputs.csv")
    rho_lookup = dsr_inputs.set_index("unit").to_dict("index")
    ...
    m = pc.s4_metrics_unit(
        rets, spy_rets,
        n_trials=96,  # full grid_size
        sr_variance_annualized=rho_lookup[unit]["sr_var_full_grid"],
        rho_bar=rho_lookup[unit]["rho_bar"],
    )
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "compute_dsr_inputs",
    "compute_unit_dsr_inputs",
    "compute_dsr_inputs_ohlc",
    "compute_unit_dsr_inputs_ohlc",
]


def compute_unit_dsr_inputs(
    unit_id: str,
    cohort: str,
    row: pd.Series,
    close: pd.DataFrame,
    combos: list[dict],
    compute_pos_raw: Callable[..., np.ndarray],
    build_unit_price: Callable[[pd.Series, pd.DataFrame], pd.Series | None],
    costs_bps: float = 10.0,
    is_ratio: float = 0.8,
    annualization: int = 252,
    min_bars: int = 250,
) -> dict:
    """Compute (rho_bar, sr_var_full_grid) for ONE unit.

    Parameters
    ----------
    unit_id : str
        Display name e.g. ``"SPY"`` or ``"SPY/TLT"``.
    cohort : str
        ``"singles"`` or ``"ratios"``.
    row : pd.Series
        Universe row passed through to ``build_unit_price``. Must contain
        leg-ticker columns the price-builder expects.
    close : pd.DataFrame
        Close prices keyed by ticker (output of MDA fetch).
    combos : list[dict]
        Full S2 grid as list of param-dicts, e.g.
        ``[{"ema_lookback": 10, "atr_mult_entry": 1.5, ...}, ...]``.
    compute_pos_raw : Callable[..., np.ndarray]
        Signal kernel: ``compute_pos_raw(price_arr, **combo_params) -> pos {0,1}``.
        Must accept all keys from `combos[i]` as keyword args.
    build_unit_price : Callable
        Strategy ``common.build_unit_price`` — turns a universe row + close
        DataFrame into a per-unit price Series (or None on insufficient data).
    costs_bps : float
        Round-trip transaction cost in basis points.
    is_ratio : float
        In-sample fraction for IS-Sharpe computation (default 0.8 matches
        S2 sweep convention).
    annualization : int
        Bars per year for Sharpe annualization (252 daily).
    min_bars : int
        Minimum price-series length; below this the unit is skipped.

    Returns
    -------
    dict
        Schema columns documented in module docstring. NaNs on skip.
    """
    out = {
        "unit": unit_id.replace("/", "__"),
        "cohort": cohort,
        "rho_bar": float("nan"),
        "n_valid_combos": 0,
        "sr_var_full_grid": float("nan"),
        "is_sharpe_max_full": float("nan"),
        "grid_size": len(combos),
    }
    price_s = build_unit_price(row, close)
    if price_s is None or len(price_s) < min_bars:
        return out
    price_arr = price_s.to_numpy(dtype=np.float64)
    bar_ret = pd.Series(price_arr, index=price_s.index).pct_change().fillna(0.0).to_numpy()

    n_bars = len(price_arr)
    combo_returns = np.zeros((n_bars, len(combos)), dtype=np.float64)
    is_sharpes = np.full(len(combos), np.nan)
    valid_mask = np.zeros(len(combos), dtype=bool)
    is_end = int(n_bars * is_ratio)

    for ci, params in enumerate(combos):
        try:
            pos_raw = compute_pos_raw(price_arr, **params)
            pos = pos_raw.astype(np.float64)
            sr_arr = np.empty(n_bars, dtype=np.float64)
            sr_arr[0] = 0.0
            sr_arr[1:] = pos[:-1] * bar_ret[1:]
            trans = np.zeros(n_bars, dtype=np.float64)
            trans[1:] = np.abs(pos[1:] - pos[:-1])
            sr_arr = sr_arr - trans * (costs_bps / 10000.0)
            combo_returns[:, ci] = sr_arr
            if sr_arr.std() > 1e-10:
                valid_mask[ci] = True
                is_slice = sr_arr[:is_end]
                if is_slice.std() > 1e-10:
                    is_sharpes[ci] = float(
                        is_slice.mean() / is_slice.std() * np.sqrt(annualization)
                    )
        except Exception:
            pass

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        out["n_valid_combos"] = int(len(valid_idx))
        return out

    valid_returns = combo_returns[:, valid_idx]
    cr = np.corrcoef(valid_returns, rowvar=False)
    n = cr.shape[0]
    upper = cr[np.triu_indices(n, k=1)]
    upper_finite = upper[np.isfinite(upper)]
    if len(upper_finite) == 0:
        out["n_valid_combos"] = int(len(valid_idx))
        return out

    out["rho_bar"] = float(upper_finite.mean())
    out["n_valid_combos"] = int(len(valid_idx))
    valid_sharpes = is_sharpes[valid_idx]
    valid_sharpes = valid_sharpes[~np.isnan(valid_sharpes)]
    if len(valid_sharpes) >= 2:
        out["sr_var_full_grid"] = float(valid_sharpes.var(ddof=1))
        out["is_sharpe_max_full"] = float(valid_sharpes.max())
    return out


def compute_dsr_inputs(
    universe: pd.DataFrame,
    close: pd.DataFrame,
    combos: list[dict],
    compute_pos_raw: Callable[..., np.ndarray],
    build_unit_price: Callable,
    costs_bps: float = 10.0,
    is_ratio: float = 0.8,
    annualization: int = 252,
    min_bars: int = 250,
    n_workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-unit (rho_bar, sr_var_full_grid) for every unit in universe.

    Parallelized via ThreadPoolExecutor (signal kernels typically njit-compiled
    and release the GIL). Single-pass over universe; one job per unit.

    Parameters
    ----------
    universe : pd.DataFrame
        Must contain a ``unit`` column and a ``cohort`` column. All other
        columns passed through to ``build_unit_price``.
    close : pd.DataFrame
        Pre-fetched Close prices for every ticker the universe needs.
    combos : list[dict]
        Full S2 grid as list of param-dicts. Each dict's keys must match the
        keyword args of ``compute_pos_raw``.
    compute_pos_raw, build_unit_price, costs_bps, is_ratio, annualization,
    min_bars : passed through to ``compute_unit_dsr_inputs``.
    n_workers : int
        Parallel workers. Set to 1 for sequential debugging.
    verbose : bool
        Print progress every 200 units + summary stats at end.

    Returns
    -------
    pd.DataFrame
        One row per universe unit. Schema in module docstring.
    """
    if verbose:
        print(
            f"compute_dsr_inputs: {len(universe)} units, "
            f"{len(combos)} combos/unit, {n_workers} workers",
            flush=True,
        )
    t0 = time.time()
    results = []

    def _one(row_tuple):
        _, row = row_tuple
        return compute_unit_dsr_inputs(
            unit_id=str(row["unit"]),
            cohort=str(row["cohort"]),
            row=row,
            close=close,
            combos=combos,
            compute_pos_raw=compute_pos_raw,
            build_unit_price=build_unit_price,
            costs_bps=costs_bps,
            is_ratio=is_ratio,
            annualization=annualization,
            min_bars=min_bars,
        )

    if n_workers <= 1:
        for rt in universe.iterrows():
            results.append(_one(rt))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_one, rt) for rt in universe.iterrows()]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if verbose and i % 200 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(futs) - i) / rate
                    print(
                        f"  [{i:4d}/{len(futs)}] elapsed={elapsed:5.1f}s "
                        f"rate={rate:5.1f}u/s ETA={eta:5.1f}s",
                        flush=True,
                    )

    df = pd.DataFrame(results)

    if verbose:
        r = df["rho_bar"].dropna()
        print(f"\nTotal elapsed: {time.time()-t0:.1f}s", flush=True)
        if len(r):
            print(
                f"rho_bar: N={len(r)}, mean={r.mean():.4f}, "
                f"median={r.median():.4f}, range=[{r.min():.4f}, {r.max():.4f}]",
                flush=True,
            )

    return df


# ---------------------------------------------------------------------------
# OHLC variants — for strategies whose signal kernel needs Close + High + Low
# (e.g., IBS, candlestick patterns). Strategy provides:
#   - `fetch_ohlc(tickers) -> dict[str, pd.DataFrame]` returning Close/High/Low
#   - `build_unit_ohlc(row, ohlc) -> tuple(index, close, high, low) | None`
#   - `compute_pos_raw(close, high, low, **params) -> np.ndarray`
# ---------------------------------------------------------------------------


def compute_unit_dsr_inputs_ohlc(
    unit_id: str,
    cohort: str,
    row: pd.Series,
    ohlc: dict,
    combos: list[dict],
    compute_pos_raw: Callable[..., np.ndarray],
    build_unit_ohlc: Callable,
    costs_bps: float = 10.0,
    is_ratio: float = 0.8,
    annualization: int = 252,
    min_bars: int = 250,
) -> dict:
    """OHLC variant of `compute_unit_dsr_inputs`.

    Same returns schema. ``build_unit_ohlc(row, ohlc)`` must return
    ``(DatetimeIndex, close, high, low)`` arrays or None on insufficient data.
    ``compute_pos_raw`` must accept positional ``(close, high, low, **params)``.
    """
    out = {
        "unit": unit_id.replace("/", "__"),
        "cohort": cohort,
        "rho_bar": float("nan"),
        "n_valid_combos": 0,
        "sr_var_full_grid": float("nan"),
        "is_sharpe_max_full": float("nan"),
        "grid_size": len(combos),
    }
    payload = build_unit_ohlc(row, ohlc)
    if payload is None:
        return out
    idx, close, high, low = payload
    if len(close) < min_bars:
        return out
    bar_ret = np.zeros(len(close), dtype=np.float64)
    bar_ret[1:] = (close[1:] / close[:-1]) - 1.0

    n_bars = len(close)
    combo_returns = np.zeros((n_bars, len(combos)), dtype=np.float64)
    is_sharpes = np.full(len(combos), np.nan)
    valid_mask = np.zeros(len(combos), dtype=bool)
    is_end = int(n_bars * is_ratio)

    for ci, params in enumerate(combos):
        try:
            pos_raw = compute_pos_raw(close, high, low, **params)
            pos = pos_raw.astype(np.float64)
            sr_arr = np.empty(n_bars, dtype=np.float64)
            sr_arr[0] = 0.0
            sr_arr[1:] = pos[:-1] * bar_ret[1:]
            trans = np.zeros(n_bars, dtype=np.float64)
            trans[1:] = np.abs(pos[1:] - pos[:-1])
            sr_arr = sr_arr - trans * (costs_bps / 10000.0)
            combo_returns[:, ci] = sr_arr
            if sr_arr.std() > 1e-10:
                valid_mask[ci] = True
                is_slice = sr_arr[:is_end]
                if is_slice.std() > 1e-10:
                    is_sharpes[ci] = float(
                        is_slice.mean() / is_slice.std() * np.sqrt(annualization)
                    )
        except Exception:
            pass

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        out["n_valid_combos"] = int(len(valid_idx))
        return out

    valid_returns = combo_returns[:, valid_idx]
    cr = np.corrcoef(valid_returns, rowvar=False)
    n = cr.shape[0]
    upper = cr[np.triu_indices(n, k=1)]
    upper_finite = upper[np.isfinite(upper)]
    if len(upper_finite) == 0:
        out["n_valid_combos"] = int(len(valid_idx))
        return out

    out["rho_bar"] = float(upper_finite.mean())
    out["n_valid_combos"] = int(len(valid_idx))
    valid_sharpes = is_sharpes[valid_idx]
    valid_sharpes = valid_sharpes[~np.isnan(valid_sharpes)]
    if len(valid_sharpes) >= 2:
        out["sr_var_full_grid"] = float(valid_sharpes.var(ddof=1))
        out["is_sharpe_max_full"] = float(valid_sharpes.max())
    return out


def compute_dsr_inputs_ohlc(
    universe: pd.DataFrame,
    ohlc: dict,
    combos: list[dict],
    compute_pos_raw: Callable[..., np.ndarray],
    build_unit_ohlc: Callable,
    costs_bps: float = 10.0,
    is_ratio: float = 0.8,
    annualization: int = 252,
    min_bars: int = 250,
    n_workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """OHLC variant of `compute_dsr_inputs`. Same schema, OHLC plumbing."""
    if verbose:
        print(
            f"compute_dsr_inputs_ohlc: {len(universe)} units, "
            f"{len(combos)} combos/unit, {n_workers} workers",
            flush=True,
        )
    t0 = time.time()
    results = []

    def _one(row_tuple):
        _, row = row_tuple
        return compute_unit_dsr_inputs_ohlc(
            unit_id=str(row["unit"]),
            cohort=str(row["cohort"]),
            row=row,
            ohlc=ohlc,
            combos=combos,
            compute_pos_raw=compute_pos_raw,
            build_unit_ohlc=build_unit_ohlc,
            costs_bps=costs_bps,
            is_ratio=is_ratio,
            annualization=annualization,
            min_bars=min_bars,
        )

    if n_workers <= 1:
        for rt in universe.iterrows():
            results.append(_one(rt))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_one, rt) for rt in universe.iterrows()]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if verbose and i % 200 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(futs) - i) / rate
                    print(
                        f"  [{i:4d}/{len(futs)}] elapsed={elapsed:5.1f}s "
                        f"rate={rate:5.1f}u/s ETA={eta:5.1f}s",
                        flush=True,
                    )

    df = pd.DataFrame(results)
    if verbose:
        r = df["rho_bar"].dropna()
        print(f"\nTotal elapsed: {time.time()-t0:.1f}s", flush=True)
        if len(r):
            print(
                f"rho_bar: N={len(r)}, mean={r.mean():.4f}, "
                f"median={r.median():.4f}, range=[{r.min():.4f}, {r.max():.4f}]",
                flush=True,
            )
    return df
