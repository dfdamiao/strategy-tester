"""Exhaustive grid search for MA window [ma_low..ma_high] step=1.

Uses Numba-compiled backtest for speed (~0.1ms per eval).
No VBT — pure numpy/numba for maximum throughput.
Supports 5 signal methods: SMA, EMA, dual MA, KAMA, momentum.

References:
    Pardo, EOTS 2e (2008) — exhaustive search when feasible
    Chan, Quantitative Trading (2008) — per-asset optimization
    Murphy, Technical Analysis (1999) Ch.9 — MA crossover
    Kaufman, TSM 6e (2020) Ch.17 — KAMA, Efficiency Ratio
    Moskowitz, Ooi & Pedersen (2012) — time-series momentum
    Degrees-of-freedom shrinkage — penalized Sharpe (analogous to adjusted-R²)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

import numba
import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage

ANNUALIZE = 252


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


# -----------------------------------------------------------------
# Numba backtest — reused from standalone stage1_screening.py
# -----------------------------------------------------------------

@numba.njit(cache=True)
def _backtest_numba(
    signal_series: np.ndarray,
    close: np.ndarray,
    n: int,
) -> tuple[float, float, float, int, float, float, float, float]:
    """Core backtest (Numba JIT).

    Returns: (sharpe, cagr, max_dd, n_trades, hit_rate,
              avg_trade_duration, profit_factor, payoff_ratio).
    Signal: signal_series[t] = 1 → in position on day t+1.

    Diagnostics added for trend quality filtering:
        avg_trade_duration — Murphy (1999) Ch.9, Pardo (2008) §6.3
        profit_factor — Pardo (2008) §6.3, Chan (2013) AT Ch.3
        payoff_ratio — Chan (2013) AT Ch.3
    """
    in_position = False
    trade_entry_ret_sum = 0.0
    trade_duration = 0
    n_trades = 0
    n_winning = 0
    total_duration = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_win = 0.0
    sum_loss = 0.0
    n_losing = 0
    daily_rets = np.zeros(n - 1)

    for t in range(n - 1):
        ret = close[t + 1] / close[t] - 1.0 if close[t] > 0 else 0.0
        sig = signal_series[t]

        if not in_position and sig == 1:
            in_position = True
            n_trades += 1
            trade_entry_ret_sum = 0.0
            trade_duration = 0

        if in_position:
            daily_rets[t] = ret
            trade_entry_ret_sum += ret
            trade_duration += 1

        if in_position and sig == 0:
            in_position = False
            total_duration += trade_duration
            if trade_entry_ret_sum > 0.0:
                n_winning += 1
                gross_profit += trade_entry_ret_sum
                sum_win += trade_entry_ret_sum
            else:
                n_losing += 1
                gross_loss += abs(trade_entry_ret_sum)
                sum_loss += abs(trade_entry_ret_sum)

    # Close open trade at end
    if in_position:
        total_duration += trade_duration
        if trade_entry_ret_sum > 0.0:
            n_winning += 1
            gross_profit += trade_entry_ret_sum
            sum_win += trade_entry_ret_sum
        else:
            n_losing += 1
            gross_loss += abs(trade_entry_ret_sum)
            sum_loss += abs(trade_entry_ret_sum)

    if n - 1 < 2:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

    mean_r = 0.0
    for r in daily_rets:
        mean_r += r
    mean_r /= float(n - 1)

    var_r = 0.0
    for r in daily_rets:
        var_r += (r - mean_r) ** 2
    var_r /= float(n - 2)
    std_r = var_r ** 0.5

    if std_r < 1e-10:
        return 0.0, 0.0, 0.0, n_trades, 0.0, 0.0, 0.0, 0.0

    sharpe = (mean_r / std_r) * (ANNUALIZE ** 0.5)
    cagr = (1.0 + mean_r) ** ANNUALIZE - 1.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_rets:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

    hit_rate = (
        float(n_winning) / float(n_trades) if n_trades > 0 else 0.0
    )
    avg_dur = float(total_duration) / float(n_trades) if n_trades > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0
    avg_win = sum_win / float(n_winning) if n_winning > 0 else 0.0
    avg_loss = sum_loss / float(n_losing) if n_losing > 0 else 0.0
    pr = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    return sharpe, cagr, max_dd, n_trades, hit_rate, avg_dur, pf, pr


# -----------------------------------------------------------------
# Signal computation functions — one per signal method
# Each returns int32 array: 1 = in position, 0 = out.
# Shifted by 1 bar (next-bar execution). min_periods=window.
# -----------------------------------------------------------------

def _signal_sma(series: np.ndarray, window: int) -> np.ndarray:
    """SMA crossover: series > SMA(window). Murphy (1999) Ch.9."""
    s = pd.Series(series)
    ma = s.rolling(window, min_periods=window).mean()
    sig = (s > ma).astype(int).values
    sig = np.roll(sig, 1)
    sig[0] = 0
    return sig.astype(np.int32)


def _signal_ema(series: np.ndarray, window: int) -> np.ndarray:
    """EMA crossover: series > EMA(window). Murphy (1999) Ch.9."""
    s = pd.Series(series)
    ema = s.ewm(span=window, min_periods=window).mean()
    sig = (s > ema).astype(int).values
    sig = np.roll(sig, 1)
    sig[0] = 0
    return sig.astype(np.int32)


def _signal_dual_ma(series: np.ndarray, window: int) -> np.ndarray:
    """Dual MA: short SMA > long SMA. Murphy (1999), Clenow (2013).

    window = long MA period. short = max(int(window/3), 5).
    """
    s = pd.Series(series)
    short_w = max(int(window / 3), 5)
    short_ma = s.rolling(short_w, min_periods=short_w).mean()
    long_ma = s.rolling(window, min_periods=window).mean()
    sig = (short_ma > long_ma).astype(int).values
    sig = np.roll(sig, 1)
    sig[0] = 0
    return sig.astype(np.int32)


@numba.njit(cache=True)
def _kama_core(
    series: np.ndarray, period: int, n: int,
) -> np.ndarray:
    """KAMA computation (Numba). Kaufman TSM 6e Ch.17."""
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    kama = np.empty(n, dtype=np.float64)
    kama[:period] = np.nan

    if period >= n:
        return kama

    kama[period] = series[period]
    for t in range(period + 1, n):
        direction = abs(series[t] - series[t - period])
        volatility = 0.0
        for j in range(t - period + 1, t + 1):
            volatility += abs(series[j] - series[j - 1])
        if volatility > 0:
            er = direction / volatility
        else:
            er = 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[t] = kama[t - 1] + sc * (series[t] - kama[t - 1])
    return kama


def _signal_kama(series: np.ndarray, window: int) -> np.ndarray:
    """KAMA crossover: series > KAMA(window). Kaufman TSM 6e Ch.17."""
    n = len(series)
    kama = _kama_core(series, window, n)
    s_arr = series.copy()
    sig = np.zeros(n, dtype=np.int32)
    for t in range(n):
        if not np.isnan(kama[t]) and s_arr[t] > kama[t]:
            sig[t] = 1
    sig = np.roll(sig, 1)
    sig[0] = 0
    return sig.astype(np.int32)


def _signal_momentum(series: np.ndarray, window: int) -> np.ndarray:
    """Momentum: N-bar return > 0. Moskowitz, Ooi & Pedersen (2012)."""
    s = pd.Series(series)
    ret = s.pct_change(window)
    sig = (ret > 0).astype(int).values
    # NaN positions → 0
    sig = np.nan_to_num(sig, nan=0).astype(int)
    sig = np.roll(sig, 1)
    sig[0] = 0
    return sig.astype(np.int32)


# Map signal_fn.__name__ → position signal function
_SIGNAL_DISPATCH: dict[str, Callable] = {
    "ma_crossover": _signal_sma,
    "ema_crossover": _signal_ema,
    "dual_ma_crossover": _signal_dual_ma,
    "kama_crossover": _signal_kama,
    "momentum": _signal_momentum,
}


def _penalized_sharpe(
    sharpe: float, n_params: int, n_trades: int,
) -> float:
    """DoF shrinkage penalty — SR × sqrt(1 - p/n), analogous to adjusted-R²."""
    if n_trades < n_params or n_trades == 0:
        return -999.0
    penalty = np.sqrt(1.0 - n_params / n_trades)
    return sharpe * penalty


# -----------------------------------------------------------------
# Per-item grid search
# -----------------------------------------------------------------

def _build_series(
    prices: dict[str, np.ndarray],
    pair_row: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (signal_base, num_close) arrays."""
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    asset_type = pair_row.get("asset_type", "pair")

    if num not in prices:
        return None

    if asset_type == "single" or num == den:
        c = prices[num]
        return c, c

    if den not in prices:
        return None

    num_c = prices[num]
    den_c = prices[den]
    min_len = min(len(num_c), len(den_c))
    if min_len < 504:
        return None
    ratio = num_c[:min_len] / den_c[:min_len]
    return ratio, num_c[:min_len]


def _process_one(
    prices: dict[str, np.ndarray],
    pair_row: dict,
    ma_low: int,
    ma_high: int,
    is_ratio: float,
    min_is_trades: int,
    min_oos_trades: int,
    signal_compute: Callable,
    signal_name: str,
    min_is_sharpe: float = 0.5,
    min_oos_sharpe: float = 0.5,
    config_min_trade_dur: float = 0.0,
) -> dict | None:
    """Grid search over all windows [ma_low..ma_high] for one item."""
    pair_name = pair_row["pair"]
    num = pair_row["numerator"]
    den = pair_row["denominator"]

    data = _build_series(prices, pair_row)
    if data is None:
        return None
    signal_base, num_close = data
    n_total = len(signal_base)

    split_idx = int(n_total * is_ratio)
    if split_idx < ma_high + 10 or (n_total - split_idx) < 60:
        return None

    is_base = signal_base[:split_idx]
    is_num = num_close[:split_idx]
    oos_base = signal_base[split_idx:]
    oos_num = num_close[split_idx:]

    # Grid search on IS — find best window by penalized Sharpe
    best_w = ma_low
    best_pen_sharpe = -999.0

    for w in range(ma_low, ma_high + 1):
        if w >= len(is_base) - 10:
            break
        sig = signal_compute(is_base, w)
        sharpe, _, _, n_trades, _, _, _, _ = _backtest_numba(
            sig, is_num, len(is_num),
        )
        if np.isnan(sharpe) or n_trades < min_is_trades:
            continue
        pen = _penalized_sharpe(sharpe, 1, n_trades)
        if pen > best_pen_sharpe:
            best_pen_sharpe = pen
            best_w = w

    if best_pen_sharpe <= -999.0:
        return None

    # IS stats with best window
    is_sig = signal_compute(is_base, best_w)
    is_sharpe, _, _, is_trades, _, _, _, _ = _backtest_numba(
        is_sig, is_num, len(is_num),
    )

    # IS Sharpe floor — reject before OOS to filter overfitting
    # (Bailey & López de Prado 2012: IS SR < 0.5 likely spurious)
    if np.isnan(is_sharpe) or is_sharpe < min_is_sharpe:
        return None

    # OOS evaluation — window fixed from IS (zero lookahead)
    # Prepend best_w bars from IS tail as warmup context so the MA is
    # fully primed at the OOS boundary.  Collect returns only from the
    # OOS portion.  Same pattern as S3 (Chan, Algorithmic Trading Ch.3).
    warmup = best_w
    warmup_base = signal_base[max(0, split_idx - warmup) : split_idx]
    ext_base = np.concatenate([warmup_base, oos_base])
    ext_sig = signal_compute(ext_base, best_w)
    # Trim warmup prefix — keep only the OOS portion
    oos_offset = len(warmup_base)
    oos_sig = ext_sig[oos_offset:]
    (oos_sharpe, oos_cagr, oos_max_dd, oos_trades, oos_hr,
     oos_avg_dur, oos_pf, oos_pr) = (
        _backtest_numba(oos_sig, oos_num, len(oos_num))
    )

    # Return autocorrelation (lag 5) on IS ratio returns — MOP (2012),
    # Jansen (2020) Ch.3: positive = trend persistence. Diagnostic only.
    is_rets = np.diff(signal_base[:split_idx])
    autocorr_5d = float("nan")
    if len(is_rets) > 10:
        mean_ret = np.mean(is_rets)
        diffs = is_rets - mean_ret
        c0 = np.sum(diffs * diffs)
        if c0 > 1e-12 and len(diffs) > 5:
            c5 = np.sum(diffs[5:] * diffs[:-5])
            autocorr_5d = c5 / c0

    # Gates — Chan QT (2009) Ch.5, Murphy (1999) Ch.9, Pardo (2008) §6.3
    min_trade_dur = config_min_trade_dur
    fail_reasons: list[str] = []
    if np.isnan(oos_sharpe):
        fail_reasons.append("oos_sr_nan")
    elif oos_sharpe < min_oos_sharpe:
        fail_reasons.append(f"oos_sr<{min_oos_sharpe}")
    if oos_trades < min_oos_trades:
        fail_reasons.append(f"trades<{min_oos_trades}")
    if oos_avg_dur < min_trade_dur:
        fail_reasons.append(f"avg_dur<{min_trade_dur:.0f}d")
    passed = len(fail_reasons) == 0

    # Halflife for interface compliance (informational)
    hl = compute_halflife(pd.Series(signal_base))

    return {
        "pair": pair_name,
        "numerator": num,
        "denominator": den,
        "halflife": round(hl, 2) if not np.isnan(hl) else float("nan"),
        "window": best_w,
        "entry_thresh": 0.0,
        "exit_thresh": 0.0,
        "stop_pct": 0.0,
        "slope_min": 0.0,
        "is_sharpe": round(float(is_sharpe), 4),
        "is_penalized_sharpe": round(best_pen_sharpe, 4),
        "is_trades": int(is_trades),
        "oos_sharpe": round(float(oos_sharpe), 4),
        "oos_trades": int(oos_trades),
        "oos_cagr": round(float(oos_cagr) * 100, 4),
        "oos_max_dd": round(float(oos_max_dd) * 100, 4),
        "oos_hit_rate": round(float(oos_hr) * 100, 2),
        "oos_avg_trade_dur": round(float(oos_avg_dur), 1),
        "oos_profit_factor": round(float(oos_pf), 4),
        "oos_payoff_ratio": round(float(oos_pr), 4),
        "autocorr_5d": round(float(autocorr_5d), 4),
        "passed": passed,
        "fail_reason": "|".join(fail_reasons) if fail_reasons else "",
        "signal_method": signal_name,
        "optim_method": "grid_ma",
    }


# -----------------------------------------------------------------
# Worker pool — globals inherited by fork
# -----------------------------------------------------------------

_worker_prices: dict[str, np.ndarray] = {}
_worker_signal_fn: Callable = _signal_sma
_worker_signal_name: str = "ma_crossover"


def _worker_init(
    prices: dict[str, np.ndarray],
    sig_fn: Callable,
    sig_name: str,
) -> None:
    global _worker_prices, _worker_signal_fn, _worker_signal_name
    _worker_prices = prices
    _worker_signal_fn = sig_fn
    _worker_signal_name = sig_name


def _worker_process(args: tuple) -> dict | None:
    (pair_row, ma_low, ma_high, is_ratio, min_is, min_oos,
     min_is_sr, min_oos_sr, min_trade_dur) = args
    return _process_one(
        _worker_prices, pair_row,
        ma_low, ma_high, is_ratio, min_is, min_oos,
        _worker_signal_fn, _worker_signal_name,
        min_is_sharpe=min_is_sr,
        min_oos_sharpe=min_oos_sr,
        config_min_trade_dur=min_trade_dur,
    )


@register_stage("s2_optimize")
def grid_ma(
    prices: pd.DataFrame,
    s1_result: pd.DataFrame,
    signal_fn=None,
    **config,
) -> pd.DataFrame:
    """Exhaustive grid search for MA window [ma_low..ma_high].

    Numba-compiled backtest — no VBT overhead.
    Supports: ma_crossover, ema_crossover, dual_ma_crossover,
    kama_crossover, momentum.

    Config keys:
        ma_low (10): minimum MA window
        ma_high (270): maximum MA window
        is_ratio (0.80): IS/OOS split
        min_is_trades (5): minimum IS trades
        min_oos_trades (5): minimum OOS trades to pass

    References:
        Pardo, EOTS 2e (2008) — exhaustive grid when feasible
        Chan, QT (2008) — per-asset window optimization
        Murphy (1999) Ch.9 — MA crossover signal
    """
    import multiprocessing as mp
    import time as _time

    ma_low = config.get("ma_low", 10)
    ma_high = config.get("ma_high", 270)
    is_ratio = config.get("is_ratio", 0.80)
    min_is_trades = config.get("min_is_trades", 5)
    min_oos_trades = config.get("min_oos_trades", 5)
    min_is_sharpe = config.get("min_is_sharpe", 0.5)
    min_oos_sharpe = config.get("min_oos_sharpe", 0.5)
    # Murphy (1999) Ch.9, Pardo (2008) §6.3: trend trades should last weeks.
    # 0.0 = disabled (backward compat); 10.0 for trend strategies.
    min_trade_dur = config.get("min_trade_duration", 0.0)

    # Resolve signal method
    sig_name = (
        signal_fn.__name__ if signal_fn else "ma_crossover"
    )
    sig_compute = _SIGNAL_DISPATCH.get(sig_name, _signal_sma)
    _log(f"Signal method: {sig_name}")

    # Warm up Numba JIT in main process BEFORE forking
    _backtest_numba(
        np.array([0, 1], dtype=np.int32),
        np.array([1.0, 1.0], dtype=np.float64),
        2,
    )
    # Also warm up KAMA core if needed
    if sig_name == "kama_crossover":
        _kama_core(np.array([1.0, 2.0, 3.0]), 2, 3)

    passed_pairs = s1_result[s1_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)
    _log(
        f"grid_ma: {n_pairs} items × {ma_high - ma_low + 1} windows "
        f"[{ma_low}..{ma_high}] ({sig_name})"
    )

    # Convert prices DataFrame → dict[ticker → np.ndarray]
    price_dict: dict[str, np.ndarray] = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) >= 504:
            price_dict[col] = np.asarray(s.values, dtype=np.float64)

    _log(f"  {len(price_dict)} tickers loaded into price dict")

    # Build worker args
    work_args = [
        (pr, ma_low, ma_high, is_ratio, min_is_trades, min_oos_trades,
         min_is_sharpe, min_oos_sharpe, min_trade_dur)
        for pr in passed_pairs
    ]

    rows: list[dict] = []
    n_pass = 0
    t_start = _time.time()

    # Fork context + pool
    ctx = mp.get_context("fork")
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    n_workers = min(n_workers, n_pairs)

    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(price_dict, sig_compute, sig_name),
    ) as pool:
        for i, result in enumerate(
            pool.imap_unordered(
                _worker_process, work_args, chunksize=8,
            )
        ):
            if result is not None:
                rows.append(result)
                if result["passed"]:
                    n_pass += 1
            if (i + 1) % 100 == 0 or i == n_pairs - 1:
                elapsed = _time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_pairs - i - 1) / rate if rate > 0 else 0
                _log(
                    f"grid_ma [{sig_name}]: {i + 1}/{n_pairs} "
                    f"({n_pass} pass) "
                    f"[{rate:.0f}/s, ETA {eta:.0f}s]"
                )

    _log(
        f"grid_ma [{sig_name}] complete: {n_pass} passed / "
        f"{len(rows)} evaluated / {n_pairs} total"
    )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "pair", "numerator", "denominator", "halflife",
            "window", "entry_thresh", "exit_thresh", "stop_pct",
            "slope_min", "is_sharpe", "is_penalized_sharpe",
            "is_trades", "oos_sharpe", "oos_trades",
            "oos_cagr", "oos_max_dd", "oos_hit_rate",
            "oos_avg_trade_dur", "oos_profit_factor",
            "oos_payoff_ratio", "autocorr_5d",
            "passed", "signal_method", "optim_method",
        ]
    )
