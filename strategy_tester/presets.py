"""Pre-built pipeline configurations.

Two preset families:
  - **STRATEGY_CLASS_PRESETS** (NEW, 2026-05-17) — shape-class defaults for
    new-strategy scaffolding (mean_reversion_equity, trend_breakout,
    multi_asset_regime, pair_cointegration). Matches MODULARITY.md §4 and
    the decision tree in LIB_REBUILD_PLAN.md §5.2.
  - **PRESETS** (legacy, debug-oriented) — named pipeline configs for
    debugging + CI smoke tests. Production runs use full combinatorial
    sweep, NOT presets (see parent CLAUDE.md).

To use a strategy-class preset:

    from strategy_tester.presets import STRATEGY_CLASS_PRESETS
    config = STRATEGY_CLASS_PRESETS["mean_reversion_equity"]()
    # then override signal / grid / universe via config.yaml
"""
from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# Strategy-class presets — yaml-shaped dicts for new-strategy scaffolding
# ---------------------------------------------------------------------------


def _mean_reversion_equity() -> dict:
    """Equity ETF mean-reversion (z-score / RSI / IBS / sma_distance).

    Citation: Chan QT Ch.3 (MR on ETFs); rules.md §3c (trend-pre-screen kill).
    """
    return {
        "stages": {
            "s1": {"method": "passthrough"},
            "s2": {
                "signal": "zscore_robust_mad",
                "optimize": {"method": "grid_search"},
                "is_ratio": 0.70,
            },
            "s3": {
                "method": "wfa_expanding",
                "n_folds": 8,
                "pass_gate": {
                    "mode": "2_and",
                    "pooled_oos_sharpe_min": 0.0,
                    "fold_win_rate_min": 0.625,
                },
            },
            "s4": {
                "methods": ["psr", "dsr"],
                "psr_threshold": 0.90,
                "dsr_threshold": 0.90,
                "use_effective_n": True,
            },
            "s5": {
                "schemes": "canonical_59",
                "benchmark": "SPY",
                "selection_metric": "ir_vs_spy",
                "apply_dsr_haircut": True,
                "cash_buffer": 0.95,
            },
        },
        "annualization": 252,
    }


def _trend_breakout() -> dict:
    """Trend/breakout (OBV pivots, trendline, MA cross) on equity ETFs.

    Citation: Pardo §6.3 (long-period detection — non-folding); rules.md §3c.
    Note: trendline_breakout / obv_pivot MUST use full-period IS/OOS at S3,
    NOT folding, because the signal spans years.
    """
    return {
        "stages": {
            "s1": {"method": "passthrough"},
            "s2": {
                "signal": "ema_crossover",
                "optimize": {"method": "grid_ma"},
                "use_s2_window": True,
                "is_ratio": 0.70,
            },
            "s3": {
                "method": "bootstrap_ci",
                "bootstrap_n": 1000,
            },
            "s4": {
                "methods": ["psr", "dsr"],
                "psr_threshold": 0.90,
                "use_effective_n": True,
            },
            "s5": {
                "schemes": "canonical_59",
                "benchmark": "SPY",
                "apply_dsr_haircut": True,
                "cash_buffer": 0.95,
            },
        },
        "annualization": 252,
    }


def _multi_asset_regime() -> dict:
    """Multi-asset regime-based (TSMOM, FRED regime, etc.).

    Citation: Antonacci 2014; LdP AFML Ch.11 (regimes).
    Note: regime strategies often have too few trades for DSR — `methods`
    starts with PSR only; add `dsr` only if trade count > 50 per unit.
    """
    return {
        "stages": {
            "s1": {"method": "passthrough"},
            "s2": {
                "signal": "regime_switch",
                "optimize": {"method": "grid_search"},
                "is_ratio": 0.70,
            },
            "s3": {"method": "wfa_expanding", "n_folds": 8},
            "s4": {
                "methods": ["psr"],
                "psr_threshold": 0.90,
            },
            "s5": {
                "schemes": "canonical_59",
                "benchmark": "60_40",
                "apply_dsr_haircut": True,
                "cash_buffer": 0.95,
            },
        },
        "annualization": 252,
    }


def _pair_cointegration() -> dict:
    """Pair / cointegration z-score strategies.

    Citation: Chan QT Ch.5; LdP AFML Ch.7 (cointegration); rules.md §3c.
    """
    return {
        "stages": {
            "s1": {"method": "chan_combined"},  # halflife + hurst composite
            "s2": {
                "signal": "zscore_robust_mad",
                "optimize": {"method": "grid_search"},
                "is_ratio": 0.70,
            },
            "s3": {
                "method": "cpcv",
                "cpcv": {
                    "k": 10,
                    "n_test": 2,
                    "purge_pct": 0.01,
                    "embargo_pct": 0.01,
                },
            },
            "s4": {
                "methods": ["psr", "dsr"],
                "psr_threshold": 0.90,
                "use_effective_n": True,
            },
            "s5": {
                "schemes": "canonical_59",
                "benchmark": "SPY",
                "apply_dsr_haircut": True,
                "cash_buffer": 0.95,
            },
        },
        "annualization": 252,
    }


STRATEGY_CLASS_PRESETS: dict[str, Callable[[], dict]] = {
    "mean_reversion_equity": _mean_reversion_equity,
    "trend_breakout": _trend_breakout,
    "multi_asset_regime": _multi_asset_regime,
    "pair_cointegration": _pair_cointegration,
}


def list_strategy_class_presets() -> list[str]:
    """Return registered strategy-class preset names."""
    return sorted(STRATEGY_CLASS_PRESETS)


def get_strategy_class_preset(name: str) -> dict:
    """Resolve a strategy-class preset by name."""
    if name not in STRATEGY_CLASS_PRESETS:
        valid = list_strategy_class_presets()
        raise KeyError(f"Unknown preset {name!r}. Valid: {valid}")
    return STRATEGY_CLASS_PRESETS[name]()


# ---------------------------------------------------------------------------
# Legacy debug presets — kept for back-compat with existing callers
# ---------------------------------------------------------------------------


PRESETS: dict[str, dict] = {
    "P-1": {
        "s1": "chan_halflife",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "chan_is_oos",
        "s4": "t_test",
        "s5": "equal_weight",
        "name": "Chan Pure",
    },
    "P-3": {
        "s1": "chan_halflife",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "wfa_expanding",
        "s4": "wfe",
        "s5": "equal_weight",
        "name": "Pardo Pure",
    },
    "P-4": {
        "s1": "chan_halflife",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "bootstrap_ci",
        "s4": "carver_2sigma",
        "s5": "handcraft_carver",
        "name": "Carver Pure",
    },
    "P-5": {
        "s1": "chan_halflife",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "monte_carlo",
        "s4": "carver_2sigma",
        "s5": "equal_weight",
        "name": "Davey Pure",
    },
    "X-1": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Chan-Prado-Carver",
    },
    "X-2": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "wfa_expanding",
        "s4": "dsr",
        "s5": "hrp",
        "name": "Chan-Pardo-Prado",
    },
    "X-4": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "monte_carlo",
        "s4": "carver_2sigma",
        "s5": "handcraft_carver",
        "name": "Chan-Davey-Carver",
    },
    "X-7": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "half_kelly",
        "name": "Chan-Prado-Thorp",
    },
    "X-11": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": ["wfa_expanding", "cpcv"],
        "s4": ["dsr", "wfe"],
        "s5": "handcraft_carver",
        "name": "Full Hybrid",
    },
    "SIG-A": {
        "s1": "chan_combined",
        "s2_signal": "zscore_robust_mad",
        "s2_optim": "grid_search",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Signal A (Robust MAD)",
    },
    "SIG-B": {
        "s1": "chan_combined",
        "s2_signal": "zscore_standard",
        "s2_optim": "grid_search",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Signal B (Standard)",
    },
    "SIG-C": {
        "s1": "chan_combined",
        "s2_signal": "kalman_hedge",
        "s2_optim": "grid_search",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Signal C (Kalman)",
    },
    "SIG-D": {
        "s1": "chan_combined",
        "s2_signal": "bollinger",
        "s2_optim": "grid_search",
        "s3": "wfa_rolling",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Signal D (Bollinger)",
    },
    # --- Trend-following presets (MA crossover) ---
    # S1 = passthrough (Murphy/Carver: no trend pre-screen)
    # S2 = grid_ma (Numba, exhaustive [15..300], Pardo 2008)
    "TRE-1": {
        "s1": "passthrough",
        "s2_signal": "ma_crossover",
        "s2_optim": "grid_ma",
        "s3": "chan_is_oos",
        "s4": "t_test",
        "s5": "equal_weight",
        "name": "Trend Chan Pure",
    },
    "TRE-2": {
        "s1": "passthrough",
        "s2_signal": "ma_crossover",
        "s2_optim": "grid_ma",
        "s3": "wfa_expanding",
        "s4": "wfe",
        "s5": "equal_weight",
        "name": "Trend Pardo Pure",
    },
    "TRE-3": {
        "s1": "passthrough",
        "s2_signal": "ma_crossover",
        "s2_optim": "grid_ma",
        "s3": "cpcv",
        "s4": "dsr",
        "s5": "handcraft_carver",
        "name": "Trend Prado-Carver",
    },
    # --- Alternative strategies presets ---
    # S3 = [wfa_expanding, cpcv] (proven DMA combo)
    # S4 = [psr, wfe, t_test] (differentially selective)
    # stop_grid includes 5-15% SL range (IBKR bracket orders)
    "ALT-RSI-S": {
        "s1": "passthrough",
        "s2_signal": "rsi_simple",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "default_window": 14,
        "window_grid": [3, 5, 7, 10, 14, 21],  # RSI period sweep
        "entry_grid": [15, 20, 25, 30, 35, 40, 45, 50],  # oversold
        "exit_grid": [40, 45, 50, 55, 60, 65, 70, 75, 80],  # overbought
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "RSI Simple (legacy-compat)",
    },
    "ALT-RSI-W": {
        "s1": "passthrough",
        "s2_signal": "rsi_wilder",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "default_window": 14,
        "window_grid": [3, 5, 7, 10, 14, 21],  # RSI period sweep
        "entry_grid": [15, 20, 25, 30, 35, 40, 45, 50],  # oversold
        "exit_grid": [40, 45, 50, 55, 60, 65, 70, 75, 80],  # overbought
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "RSI Wilder (true 1978)",
    },
    "ALT-VAZ": {
        "s1": "chan_halflife",
        "s2_signal": "vol_adaptive_zscore",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "Vol-Adaptive Z-Score (Sinclair/Carver)",
    },
    "ALT-ADX": {
        "s1": "chan_halflife",
        "s2_signal": "adx_regime_gate",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "ADX-Gated Z-Score (Narang/Murphy)",
    },
    "ALT-DON": {
        "s1": "passthrough",
        "s2_signal": "donchian",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "entry_grid": [15.0, 20.0, 30.0, 40.0, 55.0],
        "exit_grid": [5.0, 10.0, 15.0, 20.0],
        "stop_grid": [0.0],  # ATR stop built into signal
        "slope_grid": [0.0],
        "name": "Donchian Breakout (Turtle/Faith)",
    },
    "ALT-ATR": {
        "s1": "passthrough",
        "s2_signal": "atr_pullback",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "entry_grid": [1.0, 1.5, 2.0, 2.5, 3.0],
        "exit_grid": [0.0, 0.25, 0.5, 0.75],
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "ATR Pullback (Clenow/Murphy)",
    },
    "ALT-VFG": {
        "s1": "chan_halflife",
        "s2_signal": "vol_forecast_gate",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "Vol-Forecast Gate (Sinclair/Jansen)",
    },
    "ALT-HTD": {
        "s1": "chan_halflife",
        "s2_signal": "halflife_time_decay",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "Halflife Time-Decay (Chan/LdP)",
    },
    "ALT-COINT": {
        "s1": "chan_halflife",
        "s2_signal": "cointegration_spread",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],
        "name": "Cointegration Spread (Chan/Engle-Granger)",
    },
    "ALT-REG": {
        "s1": "passthrough",
        "s2_signal": "regime_switch",
        "s2_optim": "grid_search",
        "s3": [
            "wfa_expanding", "wfa_rolling", "cpcv",
            "bootstrap_ci", "monte_carlo", "sensitivity",
        ],
        "s4": ["psr", "wfe", "t_test"],
        "s5": "dynamic_1k",
        "stop_grid": [0.0, 0.05, 0.08, 0.10, 0.15],
        "slope_grid": [0.0],  # regime_switch uses ADX, not slope filter
        "name": "Regime Switch (Narang/Murphy/Jansen)",
    },
    # --- Ratio breakout (trendline support/resistance + fixed stop) ---
    # Rebuilt 2026-04-18: OLD-production walker + non-folding stats.
    # S1 = passthrough (Murphy 1999)
    # S2 = grid_breakout (full-period IS/OOS, Pardo §5.2; 75-combo grid)
    # S3A = neighborhood (Pardo §6.3) + S3B = bootstrap CI (Chan 2013 Ch.3)
    # S4 = [psr, t_test, dsr] (Bailey & LdP 2012/2014, Harvey-Liu-Zhu 2016)
    # S5 = dynamic_1k via Approach B (external signal-driven build_portfolio)
    "RB-PROD": {
        # 2026-04-18 rebuild: OLD-production walker mechanics (same-bar close
        # fill, theoretical stop + slippage, tx costs in walker) with
        # non-folding statistical validation (neighborhood + bootstrap + DSR).
        # Folds are inappropriate for trendline strategies whose lines span
        # years — see ratio_breakout/CLAUDE.md for rationale.
        "s1": "passthrough",
        "s2_signal": "trendline_breakout",
        "s2_optim": "grid_breakout",            # was grid_breakout_wfa (WFA dropped)
        "s3": ["neighborhood", "bootstrap"],    # was ["cpcv"] — Pardo §6.3 + Chan 2013 Ch.3
        "s4": ["psr", "t_test", "dsr"],         # + Deflated Sharpe (Bailey & LdP 2014)
        "s5": "dynamic_1k",
        "use_s2_window": True,
        "entry_confs": [3, 4, 5, 6, 7],
        "exit_confs": [3, 4, 5, 6, 7],
        "stop_pcts": [0.0, 0.05, 0.10],         # 0 = Murphy pure; 5/10% disaster stops
        "tx_cost_per_side": 0.001,              # 10 bps per side in walker (Chan AT Ch.3)
        "stop_slippage_bps": 5.0,               # 5 bps on stop fills (liquid-ETF default)
        "is_ratio": 0.70,                       # 70/30 IS/OOS (Pardo §5.2)
        "min_points": 4,
        "max_error_pct": 2.0,
        "min_period_days": 21,
        "atr_period": 14,                       # vestigial — walker ignores
        "name": "Ratio Breakout Production (IS/OOS + neighborhood + bootstrap + DSR)",
    },
    # --- OBV-pivot long-only single-asset timing ---
    # S1 = passthrough (Murphy/Chan/Carver — trend/structure signal)
    # S2 = grid_obv_pivot (162-combo: 3 methods × 3 params × 3 mappings × 6 stops)
    # S3+ deferred until S2 results are reviewed (see
    # docs/superpowers/specs/2026-04-23-obv-pivot-s1-s2-design.md).
    "OBV-PROD": {
        "s1": "passthrough",
        "s2_signal": "obv_pivot",
        "s2_optim": "grid_obv_pivot",
        "s3": None,                             # deferred — set after S2 review
        "s4": None,
        "s5": None,
        "use_s2_window": True,
        "tx_cost_per_side": 0.001,              # 10 bps/side (Chan AT 2013 Ch.3)
        "stop_slippage_bps": 5.0,               # 5 bps on stop + signal fills
        "is_ratio": 0.70,                       # 70/30 IS/OOS (Pardo §5.2)
        "atr_period": 14,                       # Wilder 1978
        "n_compare": 1,                         # HH/LH/HL/LL vs prev 1 same-type pivot
        "name": "OBV Pivot Long-Only (S1+S2 design 2026-04-23)",
    },
}
