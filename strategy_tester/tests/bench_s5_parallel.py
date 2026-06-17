"""Benchmark run_all_schemes: workers=1 vs workers=N.

Synthetic cache that mirrors a real 200-unit cohort, ~4000 bars (15yr daily).
Run from repo root:
    python \
        strategy_tester/tests/bench_s5_parallel.py
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

# silence runner log noise during bench
logging.basicConfig(level=logging.WARNING)

from strategy_tester.s5_replay.oracles import (  # noqa: E402
    all_scheme_names,
    equal_weight_map,
    hrp_map,
    inv_vol_map,
    sharpe_weighted_map,
)
from strategy_tester.s5_replay.runner import run_all_schemes  # noqa: E402

RNG = np.random.default_rng(42)
N_UNITS = 150          # realistic cohort size
N_BARS = 4_000         # ~16yr daily
SEED_NAV = 100_000.0
CASH_BUFFER = 0.95
OOS_RATIO = 0.8


def _make_cache(n_units: int, n_bars: int) -> dict[str, dict]:
    idx = pd.bdate_range("2009-01-02", periods=n_bars)
    cache: dict[str, dict] = {}
    for i in range(n_units):
        log_ret = RNG.normal(0.0003, 0.012, n_bars)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        low = close * (1 - RNG.uniform(0.0, 0.015, n_bars))
        atr = close * RNG.uniform(0.008, 0.025, n_bars)
        # ~30% signal-on rate
        pos_raw = (RNG.random(n_bars) < 0.30).astype(np.int8)
        cache[f"T{i:03d}"] = dict(
            index=idx,
            close=close,
            low=low,
            atr=atr,
            pos_raw=pos_raw,
            stop_code=0,
            stop_param=0.0,
            base_weight=1.0 / n_units,
        )
    return cache


def _make_rolling(cache: dict) -> pd.DataFrame:
    tickers = list(cache.keys())
    n = len(cache[tickers[0]]["index"])
    data = {t: RNG.normal(0.5, 0.3, n) for t in tickers}
    return pd.DataFrame(data, index=cache[tickers[0]]["index"])


def _make_spy(cache: dict) -> pd.Series:
    tickers = list(cache.keys())
    idx = cache[tickers[0]]["index"]
    r = RNG.normal(0.0003, 0.010, len(idx))
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name="SPY")


def main() -> None:
    sequential_baseline = 214.0  # measured: workers=1
    n_workers_list = [8]

    print(f"Building synthetic cache: {N_UNITS} units × {N_BARS} bars ...", flush=True)
    t0 = time.time()
    cache = _make_cache(N_UNITS, N_BARS)
    tickers = list(cache.keys())
    idx = cache[tickers[0]]["index"]
    oos_start = idx[int(len(idx) * OOS_RATIO)]
    spy_eq = _make_spy(cache)
    rolling_sr = _make_rolling(cache)
    rolling_iv = _make_rolling(cache)
    rolling_sortino = _make_rolling(cache)
    rolling_calmar = _make_rolling(cache)
    rolling_mom = _make_rolling(cache)
    rolling_er = _make_rolling(cache)

    # build returns_df for sharpe/inv_vol/hrp maps
    returns_df = rolling_sr  # proxy: same shape, positive values don't matter

    schemes = all_scheme_names()
    base_maps = {
        "equal": equal_weight_map(tickers),
        "sharpe": sharpe_weighted_map(returns_df),
        "sharpe_wt": sharpe_weighted_map(returns_df),
        "inv_vol": inv_vol_map(returns_df),
        "hrp": hrp_map(returns_df),
    }
    print(f"  Setup done in {time.time()-t0:.1f}s. "
          f"Running {len(schemes)} schemes.", flush=True)

    for workers in n_workers_list:
        t1 = time.time()
        results = run_all_schemes(
            schemes, cache, base_maps, rolling_sr,
            seed_nav=SEED_NAV, buffer=CASH_BUFFER,
            spy_eq=spy_eq, oos_start=oos_start,
            sizing_rule="cash_fraction",
            per_name_cap_default=0.10,
            dd_throttle="off",
            rolling_iv=rolling_iv,
            rolling_sortino=rolling_sortino,
            rolling_calmar=rolling_calmar,
            rolling_mom=rolling_mom,
            rolling_er=rolling_er,
            workers=workers,
        )
        elapsed = time.time() - t1
        speedup = sequential_baseline / elapsed
        print(
            f"  workers={workers}  time={elapsed:.1f}s  "
            f"speedup={speedup:.2f}×  (baseline={sequential_baseline:.0f}s)  "
            f"results={len(results)}/{len(schemes)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
