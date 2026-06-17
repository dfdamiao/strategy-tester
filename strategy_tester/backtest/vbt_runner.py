"""VBT backtest engine and signal helpers.

All functions are stateless — the caller provides the data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings(
    "ignore", message="invalid value encountered",
    category=RuntimeWarning,
)


def _import_vbt():
    """Import the stock vectorbt PyPI package."""
    import vectorbt as vbt
    return vbt


def compute_ratio(
    prices: pd.DataFrame, pair_row: dict,
) -> tuple[pd.Series, pd.Series, pd.Index]:
    """Compute ratio + num_prices for a pair, handling singles correctly.

    Singles (num == den or asset_type == 'single'): ratio = price itself.
    Pairs: ratio = price(num) / price(den).

    Returns (ratio, num_prices, common_index).
    """
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    asset_type = pair_row.get("asset_type", "pair")

    if num not in prices.columns:
        raise KeyError(f"{num} not in prices")

    if asset_type == "single" or num == den:
        series = prices[num].dropna()
        return series, series, series.index

    if den not in prices.columns:
        raise KeyError(f"{den} not in prices")

    common = prices[num].dropna().index.intersection(
        prices[den].dropna().index
    )
    ratio = prices[num].loc[common] / prices[den].loc[common]
    return ratio, prices[num].loc[common], common


def compute_halflife(ratio: pd.Series) -> float:
    """
    AR(1) OLS halflife estimate.
    delta(t) = a + b * ratio(t-1) + e → halflife = -log(2) / log(1 + b)
    Reference: Chan, Algorithmic Trading (2013) Ch.2
    """
    delta = ratio.diff().dropna()
    lagged = ratio.shift(1).dropna()
    idx = delta.index.intersection(lagged.index)
    if len(idx) < 30:
        return float("nan")
    try:
        result = stats.linregress(lagged.loc[idx].values, delta.loc[idx].values)
        slope = float(result.slope)
        if slope >= 0:
            return float("nan")
        return float(-np.log(2) / np.log(1 + slope))
    except Exception:
        return float("nan")


def robust_zscore(ratio: pd.Series, window: int) -> pd.Series:
    """
    Robust z-score: (ratio - median) / (1.4826 * MAD).
    min_periods=window: no partial windows.
    Reference: Isichenko, Quantitative Portfolio Management (2021) Ch.3
    """
    med = ratio.rolling(window, min_periods=window).median()
    mad = (ratio - med).abs().rolling(window, min_periods=window).median()
    return (ratio - med) / (1.4826 * mad.replace(0.0, float("nan")))


def zscore_slope(zscore: pd.Series, n: int = 2) -> pd.Series:
    """2-day linear slope of z-score (Chan falling-knife filter)."""
    return zscore.diff(n) / n


def calculate_penalized_sharpe(
    sharpe: float, n_params: int, n_trades: int,
) -> float:
    """Penalized Sharpe: SR × sqrt(1 - n_params/n_trades). DoF shrinkage."""
    if n_trades < n_params or n_trades == 0:
        return 0.0
    penalty = np.sqrt(1.0 - n_params / n_trades)
    return sharpe * penalty


def build_is_oos_split(
    common_idx: pd.Index,
    ratio: float = 0.80,
) -> tuple[pd.Index, pd.Index]:
    """Split index into IS and OOS portions by date."""
    n = len(common_idx)
    split = int(n * ratio)
    return common_idx[:split], common_idx[split:]


def backtest_vbt_fold(
    num_prices: pd.Series,
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    stop_pct: float = 0.0,
    slope_min: float = 0.0,
    slope_window: int = 2,
    fees: float = 0.001,
    init_cash: float = 100_000,
    signal_fn=None,
) -> dict:
    """
    Run VBT backtest on provided data.

    Returns dict with: sharpe, cagr, max_dd, n_trades, hit_rate,
    returns (pd.Series).

    Args:
        signal_fn: Optional callable(ratio, window, entry_thresh,
            exit_thresh, slope_min, slope_window) -> (entries, exits).
            If None, uses default robust z-score (Isichenko QPM Ch.3).

    Bias guards:
    - z-score uses min_periods=window
    - Signals shifted +1 bar: entry at next-bar open
    - fees applied per side
    - sl_stop: percentage-based stop-loss (0.0 = disabled)
    """
    vbt = _import_vbt()

    if signal_fn is not None:
        # Use caller-provided signal method
        entries, exits = signal_fn(
            ratio, window, entry_thresh, exit_thresh,
            slope_min=slope_min, slope_window=slope_window,
        )
    else:
        # Default: robust z-score (backward compatible)
        z = robust_zscore(ratio, window)
        slope = zscore_slope(z, slope_window)
        entries = (z <= entry_thresh) & (slope >= slope_min)
        exits = z >= exit_thresh
        # Next-bar execution
        entries = entries.shift(1, fill_value=False)
        exits = exits.shift(1, fill_value=False)

    sl_kwargs = {"sl_stop": stop_pct} if stop_pct > 0 else {}

    pf = vbt.Portfolio.from_signals(
        close=num_prices,
        entries=entries,
        exits=exits,
        fees=fees,
        freq="1D",
        init_cash=init_cash,
        direction="longonly",
        **sl_kwargs,
    )

    n_trades = int(pf.trades.count())
    hit_rate = float("nan")
    if n_trades > 0:
        hit_rate = float(pf.trades.win_rate())

    sharpe = float(pf.sharpe_ratio())
    ann_ret = float(pf.annualized_return())
    max_dd = float(pf.max_drawdown())
    returns = pf.returns()

    return {
        "sharpe": round(sharpe, 4) if not pd.isna(sharpe) else float("nan"),
        "cagr": round(ann_ret * 100, 4) if not pd.isna(ann_ret) else float("nan"),
        "max_dd": round(max_dd * 100, 4) if not pd.isna(max_dd) else float("nan"),
        "n_trades": n_trades,
        "hit_rate": (
            round(hit_rate * 100, 2) if not pd.isna(hit_rate) else float("nan")
        ),
        "returns": returns,
    }


def backtest_numba_fold(
    num_prices: pd.Series,
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    stop_pct: float = 0.0,
    slope_min: float = 0.0,
    slope_window: int = 2,
    fees: float = 0.001,
    init_cash: float = 100_000,
    signal_fn=None,
) -> dict:
    """Fast Numba backtest — drop-in replacement for backtest_vbt_fold.

    Uses grid_ma's _backtest_numba instead of VBT. ~10-50x faster.
    Only works for MA crossover signals (entry/exit = 0, no stop-loss).
    Falls back to backtest_vbt_fold for z-score / complex signals.

    Returns same dict format as backtest_vbt_fold.
    """
    # Only use fast path for MA crossover (no stop, no slope filter)
    is_ma_crossover = (
        entry_thresh == 0.0
        and exit_thresh == 0.0
        and stop_pct == 0.0
        and signal_fn is not None
    )
    if not is_ma_crossover:
        return backtest_vbt_fold(
            num_prices, ratio, window, entry_thresh, exit_thresh,
            stop_pct=stop_pct, slope_min=slope_min,
            slope_window=slope_window, fees=fees,
            init_cash=init_cash, signal_fn=signal_fn,
        )

    from strategy_tester.s2_optimize.grid_ma import _backtest_numba

    # Generate signal via signal_fn
    entries, exits = signal_fn(
        ratio, window, entry_thresh, exit_thresh,
        slope_min=slope_min, slope_window=slope_window,
    )
    # Convert to int array (1 = in position, 0 = out)
    signal_arr = entries.astype(int).values.astype(np.int32)

    close_arr = num_prices.values.astype(np.float64)
    n = len(close_arr)

    if n < 2:
        return {
            "sharpe": float("nan"), "cagr": float("nan"),
            "max_dd": float("nan"), "n_trades": 0,
            "hit_rate": float("nan"),
            "returns": pd.Series(dtype=float),
        }

    sharpe, cagr, max_dd, n_trades, hit_rate, _, _, _ = _backtest_numba(
        signal_arr, close_arr, n,
    )

    # Build daily returns array (for downstream metrics)
    daily_rets = np.zeros(n - 1)
    for t in range(n - 1):
        ret = close_arr[t + 1] / close_arr[t] - 1.0 if close_arr[t] > 0 else 0.0
        if signal_arr[t] == 1:
            daily_rets[t] = ret
    returns = pd.Series(daily_rets, index=num_prices.index[1:])

    return {
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else float("nan"),
        "cagr": round(cagr * 100, 4) if not np.isnan(cagr) else float("nan"),
        "max_dd": round(max_dd * 100, 4) if not np.isnan(max_dd) else float("nan"),
        "n_trades": n_trades,
        "hit_rate": (
            round(hit_rate * 100, 2) if not np.isnan(hit_rate) else float("nan")
        ),
        "returns": returns,
    }


def backtest_vbt_precomputed(
    num_prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    stop_pct: float = 0.0,
    fees: float = 0.001,
    init_cash: float = 100_000,
) -> dict:
    """Run VBT backtest with pre-computed entry/exit signals.

    Same as backtest_vbt_fold but skips signal generation entirely.
    Used by grid_search / optuna_tpe inner loop for speed.
    """
    vbt = _import_vbt()

    sl_kwargs = {"sl_stop": stop_pct} if stop_pct > 0 else {}

    pf = vbt.Portfolio.from_signals(
        close=num_prices,
        entries=entries,
        exits=exits,
        fees=fees,
        freq="1D",
        init_cash=init_cash,
        direction="longonly",
        **sl_kwargs,
    )

    n_trades = int(pf.trades.count())
    hit_rate = float("nan")
    if n_trades > 0:
        hit_rate = float(pf.trades.win_rate())

    sharpe = float(pf.sharpe_ratio())
    ann_ret = float(pf.annualized_return())
    max_dd = float(pf.max_drawdown())
    returns = pf.returns()

    return {
        "sharpe": round(sharpe, 4) if not pd.isna(sharpe) else float("nan"),
        "cagr": round(ann_ret * 100, 4) if not pd.isna(ann_ret) else float("nan"),
        "max_dd": round(max_dd * 100, 4) if not pd.isna(max_dd) else float("nan"),
        "n_trades": n_trades,
        "hit_rate": (
            round(hit_rate * 100, 2) if not pd.isna(hit_rate) else float("nan")
        ),
        "returns": returns,
    }


# ── Numba fast backtest for threshold signals ─────────
try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _backtest_entries_exits_numba(
        entries: np.ndarray,
        exits: np.ndarray,
        close: np.ndarray,
        stop_pct: float,
        fee: float,
    ) -> tuple[float, int, float]:
        """Numba backtest from boolean entry/exit arrays.

        ~50x faster than VBT Portfolio.from_signals().
        Returns (sharpe, n_trades, hit_rate).
        """
        n = len(close)
        daily_rets = np.zeros(n)
        in_pos = False
        entry_price = 0.0
        n_trades = 0
        n_winning = 0
        trade_ret = 0.0

        for t in range(1, n):
            ret = (
                close[t] / close[t - 1] - 1.0
                if close[t - 1] > 0 else 0.0
            )

            if in_pos and stop_pct > 0.0:
                pnl_pct = close[t] / entry_price - 1.0
                if pnl_pct <= -stop_pct:
                    daily_rets[t] = ret - fee
                    trade_ret += ret - fee
                    if trade_ret > 0.0:
                        n_winning += 1
                    in_pos = False
                    continue

            if in_pos and exits[t]:
                daily_rets[t] = ret - fee
                trade_ret += ret - fee
                if trade_ret > 0.0:
                    n_winning += 1
                in_pos = False
                continue

            if in_pos:
                daily_rets[t] = ret
                trade_ret += ret
                continue

            if entries[t] and not in_pos:
                in_pos = True
                entry_price = close[t]
                n_trades += 1
                trade_ret = -fee
                daily_rets[t] = -fee

        if in_pos and trade_ret > 0.0:
            n_winning += 1

        if n < 3:
            return 0.0, 0, 0.0

        mean_r = 0.0
        for i in range(n):
            mean_r += daily_rets[i]
        mean_r /= float(n)

        var_r = 0.0
        for i in range(n):
            var_r += (daily_rets[i] - mean_r) ** 2
        var_r /= float(n - 1)
        std_r = var_r ** 0.5

        if std_r < 1e-10:
            return 0.0, n_trades, 0.0

        sharpe = (mean_r / std_r) * (252.0 ** 0.5)
        hit_rate = (
            n_winning / n_trades if n_trades > 0 else 0.0
        )
        return sharpe, n_trades, hit_rate

    @_nb.njit(cache=True)
    def _grid_sweep_threshold_numba(
        signal_arr: np.ndarray,
        close: np.ndarray,
        entry_grid: np.ndarray,
        exit_grid: np.ndarray,
        stop_grid: np.ndarray,
        fee: float,
        min_is_trades: int,
        n_params: int,
        is_oversold: bool,
    ) -> tuple[float, float, int, int, int, int]:
        """Sweep all entry×exit×stop combos in one Numba call.

        Returns (best_pen_sharpe, best_sharpe, best_n_trades,
                 best_entry_idx, best_exit_idx, best_stop_idx).
        """
        n = len(close)
        best_pen = -1e30
        best_sharpe = 0.0
        best_trades = 0
        best_ei = 0
        best_xi = 0
        best_si = 0

        for ei in range(len(entry_grid)):
            entry_t = entry_grid[ei]
            for xi in range(len(exit_grid)):
                exit_t = exit_grid[xi]
                for si in range(len(stop_grid)):
                    stop_pct = stop_grid[si]

                    # Backtest this combo
                    daily_rets = np.zeros(n)
                    in_pos = False
                    entry_price = 0.0
                    n_trades = 0
                    n_winning = 0
                    trade_ret = 0.0

                    for t in range(1, n):
                        prev_close = close[t - 1]
                        ret = (
                            close[t] / prev_close - 1.0
                            if prev_close > 0.0 else 0.0
                        )
                        # Signal values (shifted by 1 for lookahead)
                        sig = signal_arr[t - 1]

                        if in_pos:
                            # Stop-loss check
                            if stop_pct > 0.0:
                                pnl_pct = (
                                    close[t] / entry_price - 1.0
                                )
                                if pnl_pct <= -stop_pct:
                                    daily_rets[t] = ret - fee
                                    trade_ret += ret - fee
                                    if trade_ret > 0.0:
                                        n_winning += 1
                                    in_pos = False
                                    continue

                            # Exit check
                            if is_oversold:
                                do_exit = sig >= exit_t
                            else:
                                do_exit = sig >= exit_t

                            if do_exit:
                                daily_rets[t] = ret - fee
                                trade_ret += ret - fee
                                if trade_ret > 0.0:
                                    n_winning += 1
                                in_pos = False
                                continue

                            daily_rets[t] = ret
                            trade_ret += ret
                        else:
                            # Entry check
                            if is_oversold:
                                do_entry = sig <= entry_t
                            else:
                                do_entry = sig <= -entry_t

                            if do_entry:
                                in_pos = True
                                entry_price = close[t]
                                n_trades += 1
                                trade_ret = -fee
                                daily_rets[t] = -fee

                    if in_pos and trade_ret > 0.0:
                        n_winning += 1

                    if n_trades < min_is_trades:
                        continue

                    # Sharpe
                    mean_r = 0.0
                    for i in range(n):
                        mean_r += daily_rets[i]
                    mean_r /= float(n)

                    var_r = 0.0
                    for i in range(n):
                        var_r += (daily_rets[i] - mean_r) ** 2
                    var_r /= float(n - 1)
                    std_r = var_r ** 0.5

                    if std_r < 1e-10:
                        continue

                    sharpe = (mean_r / std_r) * (252.0 ** 0.5)
                    # Penalized sharpe
                    pen = sharpe * (
                        1.0 - float(n_params) / float(n_trades)
                    ) ** 0.5

                    if pen > best_pen:
                        best_pen = pen
                        best_sharpe = sharpe
                        best_trades = n_trades
                        best_ei = ei
                        best_xi = xi
                        best_si = si

        return (
            best_pen, best_sharpe, best_trades,
            best_ei, best_xi, best_si,
        )

    @_nb.njit(cache=True)
    def _grid_sweep_regime_numba(
        z_arr: np.ndarray,
        ma_signal_arr: np.ndarray,
        adx_arr: np.ndarray,
        close: np.ndarray,
        entry_grid: np.ndarray,
        exit_grid: np.ndarray,
        stop_grid: np.ndarray,
        fee: float,
        min_is_trades: int,
        n_params: int,
        adx_trend: float,
        adx_range: float,
    ) -> tuple[float, float, int, int, int, int]:
        """Batched Numba sweep for regime_switch signal.

        MR mode (ADX < adx_range): z <= entry → buy, z >= exit → sell
        Trend mode (ADX > adx_trend): MA 0→1 → buy, 1→0 → sell
        Ambiguous: no signal.
        """
        n = len(close)
        best_pen = -1e30
        best_sharpe = 0.0
        best_trades = 0
        best_ei = 0
        best_xi = 0
        best_si = 0

        for ei in range(len(entry_grid)):
            entry_t = entry_grid[ei]
            for xi in range(len(exit_grid)):
                exit_t = exit_grid[xi]
                for si in range(len(stop_grid)):
                    stop_pct = stop_grid[si]

                    daily_rets = np.zeros(n)
                    in_pos = False
                    entry_price = 0.0
                    n_trades = 0
                    n_winning = 0
                    trade_ret = 0.0

                    for t in range(1, n):
                        prev_close = close[t - 1]
                        ret = (
                            close[t] / prev_close - 1.0
                            if prev_close > 0.0 else 0.0
                        )
                        # Signals from previous bar (shift=1)
                        z_prev = z_arr[t - 1]
                        adx_prev = adx_arr[t - 1]
                        ma_prev = ma_signal_arr[t - 1]
                        ma_prev2 = (
                            ma_signal_arr[t - 2] if t >= 2 else 0.0
                        )

                        is_ranging = adx_prev < adx_range
                        is_trending = adx_prev > adx_trend

                        # Entry signals
                        mr_entry = (
                            is_ranging and z_prev <= -entry_t
                        )
                        trend_entry = (
                            is_trending
                            and ma_prev == 1.0
                            and ma_prev2 == 0.0
                        )
                        # Exit signals
                        mr_exit = is_ranging and z_prev >= exit_t
                        trend_exit = (
                            is_trending
                            and ma_prev == 0.0
                            and ma_prev2 == 1.0
                        )

                        do_entry = mr_entry or trend_entry
                        do_exit = mr_exit or trend_exit

                        if in_pos:
                            if stop_pct > 0.0:
                                pnl_pct = (
                                    close[t] / entry_price - 1.0
                                )
                                if pnl_pct <= -stop_pct:
                                    daily_rets[t] = ret - fee
                                    trade_ret += ret - fee
                                    if trade_ret > 0.0:
                                        n_winning += 1
                                    in_pos = False
                                    continue

                            if do_exit:
                                daily_rets[t] = ret - fee
                                trade_ret += ret - fee
                                if trade_ret > 0.0:
                                    n_winning += 1
                                in_pos = False
                                continue

                            daily_rets[t] = ret
                            trade_ret += ret
                        else:
                            if do_entry:
                                in_pos = True
                                entry_price = close[t]
                                n_trades += 1
                                trade_ret = -fee
                                daily_rets[t] = -fee

                    if in_pos and trade_ret > 0.0:
                        n_winning += 1

                    if n_trades < min_is_trades:
                        continue

                    mean_r = 0.0
                    for i in range(n):
                        mean_r += daily_rets[i]
                    mean_r /= float(n)

                    var_r = 0.0
                    for i in range(n):
                        var_r += (daily_rets[i] - mean_r) ** 2
                    var_r /= float(n - 1)
                    std_r = var_r ** 0.5

                    if std_r < 1e-10:
                        continue

                    sharpe = (mean_r / std_r) * (252.0 ** 0.5)
                    pen = sharpe * (
                        1.0 - float(n_params) / float(n_trades)
                    ) ** 0.5

                    if pen > best_pen:
                        best_pen = pen
                        best_sharpe = sharpe
                        best_trades = n_trades
                        best_ei = ei
                        best_xi = xi
                        best_si = si

        return (
            best_pen, best_sharpe, best_trades,
            best_ei, best_xi, best_si,
        )

    HAS_NUMBA_BACKTEST = True
    HAS_NUMBA_GRID_SWEEP = True

except ImportError:
    HAS_NUMBA_BACKTEST = False
    HAS_NUMBA_GRID_SWEEP = False


def grid_sweep_threshold(
    signal_arr: np.ndarray,
    close: np.ndarray,
    entry_grid: list,
    exit_grid: list,
    stop_grid: list,
    fees: float = 0.001,
    min_is_trades: int = 3,
    n_params: int = 4,
    is_oversold: bool = True,
) -> dict | None:
    """Sweep entry×exit×stop combos in one Numba call.

    signal_arr: pre-computed signal (RSI, z-score, etc.) as numpy array.
    close: price array for P&L.
    is_oversold: True = RSI-style (entry ≤ thresh), False = z-score (entry ≤ -thresh).

    Returns dict with best params + sharpe, or None if nothing passes.
    """
    if not HAS_NUMBA_GRID_SWEEP:
        return None

    eg = np.array(entry_grid, dtype=np.float64)
    xg = np.array(exit_grid, dtype=np.float64)
    sg = np.array(stop_grid, dtype=np.float64)
    sig = signal_arr.astype(np.float64)
    cl = close.astype(np.float64)

    best_pen, best_sharpe, best_trades, ei, xi, si = (
        _grid_sweep_threshold_numba(
            sig, cl, eg, xg, sg, fees,
            min_is_trades, n_params, is_oversold,
        )
    )

    if best_pen <= -1e29:
        return None

    return {
        "sharpe": round(best_sharpe, 4),
        "n_trades": int(best_trades),
        "pen_sharpe": round(best_pen, 4),
        "entry_thresh": entry_grid[ei],
        "exit_thresh": exit_grid[xi],
        "stop_pct": stop_grid[si],
    }


def grid_sweep_regime(
    z_arr: np.ndarray,
    ma_signal_arr: np.ndarray,
    adx_arr: np.ndarray,
    close: np.ndarray,
    entry_grid: list,
    exit_grid: list,
    stop_grid: list,
    fees: float = 0.001,
    min_is_trades: int = 3,
    n_params: int = 4,
    adx_trend: float = 25.0,
    adx_range: float = 20.0,
) -> dict | None:
    """Batched Numba sweep for regime_switch signal."""
    if not HAS_NUMBA_GRID_SWEEP:
        return None

    eg = np.array(entry_grid, dtype=np.float64)
    xg = np.array(exit_grid, dtype=np.float64)
    sg = np.array(stop_grid, dtype=np.float64)

    best_pen, best_sharpe, best_trades, ei, xi, si = (
        _grid_sweep_regime_numba(
            z_arr.astype(np.float64),
            ma_signal_arr.astype(np.float64),
            adx_arr.astype(np.float64),
            close.astype(np.float64),
            eg, xg, sg, fees,
            min_is_trades, n_params,
            adx_trend, adx_range,
        )
    )

    if best_pen <= -1e29:
        return None

    return {
        "sharpe": round(best_sharpe, 4),
        "n_trades": int(best_trades),
        "pen_sharpe": round(best_pen, 4),
        "entry_thresh": entry_grid[ei],
        "exit_thresh": exit_grid[xi],
        "stop_pct": stop_grid[si],
    }


def backtest_numba_entries_exits(
    num_prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    stop_pct: float = 0.0,
    fees: float = 0.001,
) -> dict:
    """Fast Numba backtest for threshold-based signals.

    Drop-in replacement for backtest_vbt_precomputed.
    Falls back to VBT if Numba unavailable.
    """
    if not HAS_NUMBA_BACKTEST:
        return backtest_vbt_precomputed(
            num_prices, entries, exits,
            stop_pct=stop_pct, fees=fees,
        )

    close = num_prices.values.astype(np.float64)
    ent = entries.values.astype(bool)
    ext = exits.values.astype(bool)

    sharpe, n_trades, hit_rate = _backtest_entries_exits_numba(
        ent, ext, close, stop_pct, fees,
    )

    return {
        "sharpe": round(sharpe, 4),
        "n_trades": n_trades,
        "hit_rate": round(hit_rate * 100, 2),
    }
