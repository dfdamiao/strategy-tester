"""Regime-aware mean reversion filter. Jansen, ML4T 2e (2020) Ch.13.

Secondary filter applied on top of a primary S1 screen.
Fits a Gaussian Mixture Model (GMM) on SPY returns to identify
bull / bear / sideways regimes, then checks that the pair's ratio
is mean-reverting (halflife is valid) in BOTH bull AND bear.

A pair that only mean-reverts in bull markets is fragile — drawdowns
will come exactly when the broader market is already stressed.

Reference:
    Jansen, Machine Learning for Algorithmic Trading 2e (2020) Ch.13
    — GMM regime detection for conditional strategy validation.
    Lo, Adaptive Markets (2017) — regime-switching alpha.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from strategy_tester.backtest.vbt_runner import compute_halflife
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _fit_gmm_regimes(
    spy: pd.Series,
    n_components: int = 3,
) -> pd.Series:
    """Fit GMM on SPY returns, label regimes by mean return.

    Returns Series of labels: 'bear' (lowest mean), 'sideways', 'bull'.
    """
    returns = spy.pct_change().dropna()
    X = returns.values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components, random_state=42, n_init=5,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    # Map cluster IDs to regime names by ascending mean return
    means = gmm.means_.flatten()
    order = np.argsort(means)
    regime_map = {}
    names = ["bear", "sideways", "bull"][:n_components]
    for rank, cluster_id in enumerate(order):
        regime_map[cluster_id] = names[rank]

    regime_labels = pd.Series(
        [regime_map[lab] for lab in labels],
        index=returns.index,
        name="regime",
    )
    return regime_labels


def _process_one(
    prices: pd.DataFrame,
    pair: dict,
    regimes: pd.Series,
    min_hl: float,
    max_hl: float,
    min_rows: int,
    min_regime_bars: int,
) -> dict | None:
    num, den = pair["numerator"], pair["denominator"]
    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    if len(common) < min_rows:
        return None

    ratio = p_num.loc[common] / p_den.loc[common]

    # Overall halflife
    hl_all = compute_halflife(ratio)

    # Check MR in each regime
    regime_results = {}
    for regime_name in ["bull", "bear"]:
        regime_idx = regimes[regimes == regime_name].index
        pair_regime_idx = ratio.index.intersection(regime_idx)

        if len(pair_regime_idx) < min_regime_bars:
            regime_results[regime_name] = np.nan
            continue

        ratio_regime = ratio.loc[pair_regime_idx].sort_index()
        hl_regime = compute_halflife(ratio_regime)
        regime_results[regime_name] = hl_regime

    hl_bull = regime_results.get("bull", np.nan)
    hl_bear = regime_results.get("bear", np.nan)

    # Pass if MR is valid in BOTH bull AND bear
    def _valid_hl(h: float) -> bool:
        return not np.isnan(h) and min_hl <= h <= max_hl

    passed = _valid_hl(hl_bull) and _valid_hl(hl_bear)

    window = (
        min(max(int(hl_all * 0.5), 10), 252)
        if not np.isnan(hl_all)
        else 0
    )

    return {
        "pair": pair["pair"],
        "numerator": num,
        "denominator": den,
        "passed": passed,
        "halflife": (
            round(hl_all, 2) if not np.isnan(hl_all) else hl_all
        ),
        "window": window,
        "method": "regime_aware",
        "hl_bull": round(hl_bull, 2) if not np.isnan(hl_bull) else hl_bull,
        "hl_bear": round(hl_bear, 2) if not np.isnan(hl_bear) else hl_bear,
    }


@register_stage("s1")
def regime_aware(
    prices: pd.DataFrame,
    pairs: list[dict],
    **config,
) -> pd.DataFrame:
    """Filter pairs by regime-conditional mean reversion.

    Secondary filter. Fits GMM on SPY to detect bull/bear/sideways,
    then requires the pair's halflife to be valid in BOTH bull AND bear.

    Config:
        spy_ticker: str = "SPY"
        regime_min_halflife: float = 2
        regime_max_halflife: float = 756
        regime_min_bars: int = 126
            Minimum bars per regime for halflife to be meaningful.
        regime_n_components: int = 3
        min_common_rows: int = 252
    """
    spy_ticker = config.get("spy_ticker", "SPY")
    min_hl = config.get("regime_min_halflife", 2)
    max_hl = config.get("regime_max_halflife", 756)
    min_regime_bars = config.get("regime_min_bars", 126)
    n_components = config.get("regime_n_components", 3)
    min_rows = config.get("min_common_rows", 252)
    parallel = config.get("parallel", True)

    # Fit GMM on SPY once (shared across all pairs)
    if spy_ticker not in prices.columns:
        _log(f"regime_aware: {spy_ticker} not found, all pairs FAIL")
        return pd.DataFrame()

    spy = prices[spy_ticker].dropna()
    if len(spy) < 500:
        _log("regime_aware: insufficient SPY data, all pairs FAIL")
        return pd.DataFrame()

    _log(f"regime_aware: fitting GMM ({n_components} components) on {spy_ticker}...")
    regimes = _fit_gmm_regimes(spy, n_components=n_components)
    regime_counts = regimes.value_counts().to_dict()
    _log(f"regime_aware: regimes = {regime_counts}")

    n_pairs = len(pairs)
    rows: list[dict] = []

    if parallel and n_pairs > 50:
        n_workers = min(os.cpu_count() or 4, 6)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, prices, p,
                    regimes, min_hl, max_hl, min_rows, min_regime_bars,
                ): p
                for p in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 200 == 0 or i == n_pairs - 1:
                    _log(f"regime_aware: {i + 1}/{n_pairs}")
    else:
        for i, pair in enumerate(pairs):
            result = _process_one(
                prices, pair,
                regimes, min_hl, max_hl, min_rows, min_regime_bars,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0 or i == n_pairs - 1:
                _log(f"regime_aware: {i + 1}/{n_pairs}")

    return pd.DataFrame(rows)
