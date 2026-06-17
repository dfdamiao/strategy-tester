"""Exhaustive grid search for the OBV-pivot S2 parameters.

Per-ticker grid: 162 combos.

    method        fractal | topological | pip                     (3)
    method_param  fractal: swing ∈ {25,50,100}                    (3)
                  topological: top_k ∈ {15,30,60}
                  pip: k ∈ {30,60,120}
    mapping       A | B | C                                        (3)
    stop          none | fixed_10% | fixed_15% | fixed_20%         (6)
                  atr_2x | atr_3x

n_compare=1 fixed. ATR(14) Wilder.

Pipeline per ticker:
    1. Compute OBV, ATR(14) on full OHLCV history.
    2. For each of 27 (method, method_param, mapping) combos, compute the causal
       pre-stop 0/1 position series ONCE on the full window.
    3. 70/30 IS/OOS split by date index. Pivot timeline is causal (first-
       appearance day for topological/pip, confirmation lag for fractal), so
       slicing by date index introduces no look-ahead.
    4. Numba walker sweeps the 6 stop variants on IS and OOS independently,
       computes Sharpe / CAGR / MaxDD / trade count / hit rate.
    5. Emit one row per (method, method_param, mapping, stop) combo = 162 rows.

Parallelism: ``mp.get_context("fork") + Pool(initializer=_worker_init)``.
Each worker holds the OHLCV dict by fork COW. Pre-warm Numba JIT in parent
before forking.

References:
    Granville (1963) — OBV
    Bulkowski (2005), Edelsbrunner & Harer (2008), Fu et al. (2007) — pivots
    Murphy (1999) — trend signal
    Chan AT (2013) Ch.3 — 10 bps/side tx cost
    Pardo (2008) §5.2 — 70/30 IS/OOS
    Wilder (1978) — ATR(14)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from strategy_tester.s2_signal.obv_pivot import (
    _MAPPERS,
    causal_timeline,
    compute_obv,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
log = logging.getLogger(__name__)

ANNUALIZE = 252

# Grid spec
METHOD_PARAMS: dict[str, tuple[int, ...]] = {
    "fractal":     (25, 50, 100),
    "topological": (15, 30, 60),
    "pip":         (30, 60, 120),
}
MAPPINGS = ("A", "B", "C")

# Stop axis. stop_type_code: 0=none, 1=fixed_pct, 2=atr_trail.
# Each stop is (label, code, param).
STOPS: tuple[tuple[str, int, float], ...] = (
    ("none",       0, 0.0),
    ("fixed_10%",  1, 0.10),
    ("fixed_15%",  1, 0.15),
    ("fixed_20%",  1, 0.20),
    ("atr_2x",     2, 2.0),
    ("atr_3x",     2, 3.0),
)


# ---------------------------------------------------------------------------
# Numba walker
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _walk(
    close: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    pos_raw: np.ndarray,
    stop_type_code: int,
    stop_param: float,
    tx_cost: float,
    slip: float,
):
    """Walk a 0/1 position series with stop overlay.

    Returns (daily_rets, n_trades, n_winning, time_in_market, last_pos).
    daily_rets is on the close index; day-0 return is 0 by convention.

    Entry convention: same-bar close × (1+slip).
    Signal exit: same-bar close × (1-slip) with one-side tx cost booked.
    Fixed stop: intrabar if low[t] <= entry × (1-stop_param); fill at floor
        × (1-slip).
    ATR trail: close-based; trail[t] = max(trail[t-1], close[t] -
        stop_param × atr[t]). If close[t] <= trail[t], exit at trail × (1-slip).
    Stops take priority over signal in the same bar.
    Tx cost is booked once on entry day and once on exit day (so round-trip
    = 2 × tx_cost).
    """
    n = len(close)
    daily_rets = np.zeros(n, dtype=np.float64)
    in_pos = False
    entry_px = 0.0
    prev_price = 0.0
    trail = 0.0
    n_trades = 0
    n_winning = 0
    time_in_mkt = 0
    last_pos = 0

    for t in range(n):
        prev_in_pos = in_pos
        today_ret = 0.0

        if prev_in_pos:
            stop_hit = False
            stop_px = 0.0

            if stop_type_code == 1:
                floor = entry_px * (1.0 - stop_param)
                if low[t] <= floor:
                    stop_hit = True
                    stop_px = floor * (1.0 - slip)
            elif stop_type_code == 2:
                candidate = close[t] - stop_param * atr[t]
                if candidate > trail:
                    trail = candidate
                if close[t] <= trail:
                    stop_hit = True
                    stop_px = trail * (1.0 - slip)

            if stop_hit:
                today_ret = stop_px / prev_price - 1.0 - tx_cost
                if stop_px > entry_px:
                    n_winning += 1
                n_trades += 1
                time_in_mkt += 1
                in_pos = False
            elif pos_raw[t] == 0:
                exit_px = close[t] * (1.0 - slip)
                today_ret = exit_px / prev_price - 1.0 - tx_cost
                if exit_px > entry_px:
                    n_winning += 1
                n_trades += 1
                time_in_mkt += 1
                in_pos = False
            else:
                today_ret = close[t] / prev_price - 1.0
                prev_price = close[t]
                time_in_mkt += 1

        if not prev_in_pos and pos_raw[t] == 1:
            entry_px = close[t] * (1.0 + slip)
            prev_price = entry_px
            in_pos = True
            today_ret -= tx_cost
            if stop_type_code == 2:
                trail = close[t] - stop_param * atr[t]

        # End-of-window: force exit on final bar if still holding.
        if in_pos and t == n - 1:
            exit_px = close[t] * (1.0 - slip)
            # today_ret currently has the close/prev_price change (if we held
            # through the bar). Override with the full exit.
            today_ret = exit_px / prev_price - 1.0 - tx_cost
            if exit_px > entry_px:
                n_winning += 1
            n_trades += 1
            in_pos = False

        daily_rets[t] = today_ret
        last_pos = 1 if in_pos else 0

    return daily_rets, n_trades, n_winning, time_in_mkt, last_pos


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return float("nan")
    sd = float(np.nanstd(rets))
    if sd == 0.0 or np.isnan(sd):
        return float("nan")
    return float(np.nanmean(rets)) / sd * np.sqrt(ANNUALIZE)


def _cagr(rets: np.ndarray) -> float:
    if rets.size < 2:
        return float("nan")
    eq = np.nancumprod(1.0 + rets)
    n_years = rets.size / ANNUALIZE
    if n_years <= 0 or eq[-1] <= 0:
        return float("nan")
    return float(eq[-1] ** (1.0 / n_years) - 1.0)


def _maxdd(rets: np.ndarray) -> float:
    if rets.size < 2:
        return float("nan")
    eq = np.nancumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.nanmin(dd))


def _hit_rate(n_winning: int, n_trades: int) -> float:
    return float(n_winning / n_trades) if n_trades > 0 else 0.0


def _atr_wilder(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Wilder ATR(14) using SMA seed then RMA."""
    n = len(close)
    tr = np.zeros(n)
    for t in range(1, n):
        a = high[t] - low[t]
        b = abs(high[t] - close[t - 1])
        c = abs(low[t] - close[t - 1])
        tr[t] = max(a, b, c)
    atr = np.zeros(n)
    if n <= period:
        return atr
    atr[period] = np.mean(tr[1:period + 1])
    for t in range(period + 1, n):
        atr[t] = (atr[t - 1] * (period - 1) + tr[t]) / period
    return atr


# ---------------------------------------------------------------------------
# Per-ticker worker
# ---------------------------------------------------------------------------


_OHLCV: dict[str, dict[str, pd.Series]] = {}
_CACHE_DIR: str = ""
_TX_COST: float = 0.001
_SLIP: float = 0.0005
_IS_RATIO: float = 0.70
_ATR_PERIOD: int = 14


def _worker_init(
    ohlcv: dict,
    cache_dir: str,
    tx_cost: float,
    slip: float,
    is_ratio: float,
    atr_period: int,
) -> None:
    import warnings as _w
    _w.filterwarnings("ignore")
    global _OHLCV, _CACHE_DIR, _TX_COST, _SLIP, _IS_RATIO, _ATR_PERIOD
    _OHLCV = ohlcv
    _CACHE_DIR = cache_dir
    _TX_COST = tx_cost
    _SLIP = slip
    _IS_RATIO = is_ratio
    _ATR_PERIOD = atr_period


def _bh_metrics(close: np.ndarray) -> tuple[float, float, float]:
    """Buy-and-hold Sharpe / CAGR / MaxDD on a close series."""
    rets = np.diff(close) / close[:-1]
    return _sharpe(rets), _cagr(rets), _maxdd(rets)


def _ticker_worker(ticker: str) -> list[dict]:
    ckpt = Path(_CACHE_DIR) / f"{ticker}.pkl"
    if ckpt.exists():
        try:
            return pickle.loads(ckpt.read_bytes())
        except Exception:
            pass

    data = _OHLCV.get(ticker)
    if data is None:
        return []

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    n = len(close)
    if n < 504:
        return []

    close_arr = close.to_numpy(dtype=np.float64)
    high_arr = high.to_numpy(dtype=np.float64)
    low_arr = low.to_numpy(dtype=np.float64)

    atr_arr = _atr_wilder(close_arr, high_arr, low_arr, period=_ATR_PERIOD)

    split = int(_IS_RATIO * n)
    if split < 100 or n - split < 50:
        return []

    # B&H benchmarks on IS and OOS
    bh_is_sr, bh_is_cagr, _ = _bh_metrics(close_arr[:split])
    bh_oos_sr, bh_oos_cagr, _ = _bh_metrics(close_arr[split - 1:])

    rows: list[dict] = []
    idx = close.index
    is_start_str = str(idx[0])[:10]
    is_end_str = str(idx[split - 1])[:10]
    oos_start_str = str(idx[split])[:10]
    oos_end_str = str(idx[-1])[:10]

    # Compute OBV once; pivots are per (method, param) — invariant across mappings.
    obv = compute_obv(close, volume)

    for method, params in METHOD_PARAMS.items():
        for param in params:
            # Build the causal pivot timeline once per (method, param).
            try:
                timeline = causal_timeline(obv, method, param, n_compare=1)
            except Exception as exc:  # pragma: no cover
                log.warning("timeline failed %s %s p=%s: %s",
                            ticker, method, param, exc)
                continue

            for mapping in MAPPINGS:
                # Mapping only reshapes the timeline into a 0/1 position.
                position = _MAPPERS[mapping](timeline, close.index)
                pos_raw = position.to_numpy(dtype=np.int64)

                for stop_label, stop_code, stop_param in STOPS:
                    # IS walk
                    pos_is = pos_raw[:split].copy()
                    rets_is, trd_is, win_is, tim_is, _ = _walk(
                        close_arr[:split],
                        low_arr[:split],
                        atr_arr[:split],
                        pos_is,
                        stop_code, stop_param,
                        _TX_COST, _SLIP,
                    )

                    # OOS walk (starts flat; pivots already causal so pos_raw
                    # on OOS slice is the intended OOS signal).
                    pos_oos = pos_raw[split:].copy()
                    rets_oos, trd_oos, win_oos, tim_oos, _ = _walk(
                        close_arr[split:],
                        low_arr[split:],
                        atr_arr[split:],
                        pos_oos,
                        stop_code, stop_param,
                        _TX_COST, _SLIP,
                    )

                    is_sr = _sharpe(rets_is)
                    is_cagr = _cagr(rets_is)
                    is_dd = _maxdd(rets_is)
                    oos_sr = _sharpe(rets_oos)
                    oos_cagr = _cagr(rets_oos)
                    oos_dd = _maxdd(rets_oos)

                    rows.append({
                        "ticker": ticker,
                        "method": method,
                        "method_param": int(param),
                        "mapping": mapping,
                        "stop_type": stop_label,
                        "stop_code": int(stop_code),
                        "stop_param": float(stop_param),
                        "is_sharpe": is_sr,
                        "is_cagr": is_cagr,
                        "is_maxdd": is_dd,
                        "is_trades": int(trd_is),
                        "is_hit_rate": _hit_rate(win_is, trd_is),
                        "is_time_in_market": float(tim_is) / split if split else 0.0,
                        "oos_sharpe": oos_sr,
                        "oos_cagr": oos_cagr,
                        "oos_maxdd": oos_dd,
                        "oos_calmar": float(oos_cagr / abs(oos_dd))
                        if oos_dd and not np.isnan(oos_dd) else float("nan"),
                        "oos_trades": int(trd_oos),
                        "oos_hit_rate": _hit_rate(win_oos, trd_oos),
                        "oos_time_in_market": float(tim_oos) / (n - split)
                        if (n - split) else 0.0,
                        "bh_is_sharpe": bh_is_sr,
                        "bh_is_cagr": bh_is_cagr,
                        "bh_oos_sharpe": bh_oos_sr,
                        "bh_oos_cagr": bh_oos_cagr,
                        "oos_excess_sr_vs_bh": (oos_sr - bh_oos_sr)
                        if not (np.isnan(oos_sr) or np.isnan(bh_oos_sr))
                        else float("nan"),
                        "oos_excess_cagr_vs_bh": (oos_cagr - bh_oos_cagr)
                        if not (np.isnan(oos_cagr) or np.isnan(bh_oos_cagr))
                        else float("nan"),
                        "is_start": is_start_str,
                        "is_end": is_end_str,
                        "oos_start": oos_start_str,
                        "oos_end": oos_end_str,
                        "n_bars": int(n),
                    })

    # Apply minimal S2 gates: oos_sharpe > 0 AND oos_trades >= 5
    for r in rows:
        r["passed"] = bool(
            r["oos_sharpe"] > 0.0
            and not np.isnan(r["oos_sharpe"])
            and r["oos_trades"] >= 5
        )

    try:
        ckpt.write_bytes(pickle.dumps(rows))
    except Exception as exc:  # pragma: no cover
        log.warning("ckpt write failed for %s: %s", ticker, exc)

    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def grid_obv_pivot(
    ohlcv: dict[str, dict[str, pd.Series]],
    tickers: list[str],
    *,
    tx_cost_per_side: float = 0.001,
    stop_slippage_bps: float = 5.0,
    is_ratio: float = 0.70,
    atr_period: int = 14,
    cache_dir: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Run the 162-combo grid across all tickers in parallel.

    Returns a DataFrame with one row per (ticker × 162 combos). Callers split
    winners (passed=True) vs full grid downstream.
    """
    slip = stop_slippage_bps / 10000.0
    cache = cache_dir or str(Path.cwd() / "obv_grid_cache")
    Path(cache).mkdir(parents=True, exist_ok=True)

    # Pre-warm Numba JIT in parent before forking.
    _ = _walk(
        np.array([100.0, 101.0, 102.0, 101.0, 100.0]),
        np.array([99.0, 100.0, 101.0, 100.0, 99.0]),
        np.array([0.0, 0.5, 0.5, 0.5, 0.5]),
        np.array([0, 1, 1, 1, 0], dtype=np.int64),
        0, 0.0, tx_cost_per_side, slip,
    )

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    log.info("grid_obv_pivot: %d tickers, %d workers", len(tickers), n_workers)

    ctx = mp.get_context("fork")
    start = datetime.now()
    all_rows: list[dict] = []
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(ohlcv, cache, tx_cost_per_side, slip, is_ratio, atr_period),
    ) as pool:
        n_done = 0
        best_is_sr = float("-inf")
        best_oos_sr = float("-inf")
        best_is_cagr = float("-inf")
        best_oos_cagr = float("-inf")
        n_beat_bh_sr = 0
        n_beat_bh_cagr = 0
        n_passed_cells = 0
        for rows in pool.imap_unordered(_ticker_worker, tickers, chunksize=1):
            all_rows.extend(rows)
            n_done += 1

            # Per-ticker completion heartbeat (keeps stderr alive so the
            # user never stares at a blank terminal for minutes).
            ticker_name = rows[0]["ticker"] if rows else "(empty)"
            elapsed = (datetime.now() - start).total_seconds()
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(tickers) - n_done) / rate if rate > 0 else float("nan")
            log.info("  [%d/%d] %s done (%d rows) | elapsed=%.0fs rate=%.2f/s eta=%.0fs",
                     n_done, len(tickers), ticker_name, len(rows),
                     elapsed, rate, eta)

            # running-best trackers (scan only newly-arrived rows)
            for r in rows:
                s_is = r.get("is_sharpe", float("nan"))
                s_oos = r.get("oos_sharpe", float("nan"))
                c_is = r.get("is_cagr", float("nan"))
                c_oos = r.get("oos_cagr", float("nan"))
                ex_sr = r.get("oos_excess_sr_vs_bh", float("nan"))
                ex_cagr = r.get("oos_excess_cagr_vs_bh", float("nan"))
                if not np.isnan(s_is) and s_is > best_is_sr:
                    best_is_sr = float(s_is)
                if not np.isnan(s_oos) and s_oos > best_oos_sr:
                    best_oos_sr = float(s_oos)
                if not np.isnan(c_is) and c_is > best_is_cagr:
                    best_is_cagr = float(c_is)
                if not np.isnan(c_oos) and c_oos > best_oos_cagr:
                    best_oos_cagr = float(c_oos)
                if not np.isnan(ex_sr) and ex_sr > 0:
                    n_beat_bh_sr += 1
                if not np.isnan(ex_cagr) and ex_cagr > 0:
                    n_beat_bh_cagr += 1
                if r.get("passed", False):
                    n_passed_cells += 1

            if n_done % 10 == 0 or n_done == len(tickers):
                n_cells = len(all_rows)
                log.info(
                    "  ----- [%d/%d] aggregate | "
                    "best IS_SR=%.2f OOS_SR=%.2f IS_CAGR=%+.1f%% OOS_CAGR=%+.1f%% | "
                    "cells>BH_SR=%d/%d (%.1f%%) >BH_CAGR=%d/%d (%.1f%%) | passed=%d",
                    n_done, len(tickers),
                    best_is_sr if best_is_sr != float("-inf") else float("nan"),
                    best_oos_sr if best_oos_sr != float("-inf") else float("nan"),
                    best_is_cagr * 100 if best_is_cagr != float("-inf") else float("nan"),
                    best_oos_cagr * 100 if best_oos_cagr != float("-inf") else float("nan"),
                    n_beat_bh_sr, n_cells,
                    100 * n_beat_bh_sr / n_cells if n_cells else 0.0,
                    n_beat_bh_cagr, n_cells,
                    100 * n_beat_bh_cagr / n_cells if n_cells else 0.0,
                    n_passed_cells,
                )

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    return df
