"""Default configuration for pipeline library."""
from __future__ import annotations

DEFAULT_CONFIG: dict = {
    "is_ratio": 0.80,
    "cost_per_side": 0.001,
    "min_common_rows": 252,
    "min_halflife": 2,
    "max_halflife": 756,
    "min_window": 10,
    "max_window": 504,
    "adf_pvalue_threshold": 0.05,
    "hurst_threshold": 0.50,
    "er_threshold": 0.30,
    "er_window": 10,  # Kaufman TSM Ch.17: ER lookback (default 10 bars)
    "slope_window": 2,
    "entry_grid": [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0],  # Isichenko Ch.3
    "exit_grid": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "stop_grid": [0.0, 0.10, 0.20],  # Chan AT Ch.3: no stop for MR default
    "slope_grid": [0.0],  # fixed — falling-knife at 0 (Chan AT Ch.3)
    "cpcv_n_folds": 10,
    "cpcv_purge": 20,
    "cpcv_embargo": 5,
    "wfa_n_folds": 8,
    "wfa_rolling_is": 504,
    "wfa_rolling_oos": 180,
    "min_is_sharpe": 0.5,  # Bailey & LdP (2012): IS SR < 0.5 likely spurious
    "min_oos_sharpe": 0.0,  # Chan QT/AT: positive OOS sufficient; S3/S4 filter downstream
    "min_is_trades": 5,  # Chan QT (2009) Ch.3 — matches DMA/ETF zscore scripts
    "min_oos_trades": 5,  # Chan QT (2009) Ch.3 — matches DMA/ETF zscore scripts
    "s3_min_oos_sharpe": 0.0,  # Chan AT Ch.3, LdP AFML Ch.11: positive pooled OOS sufficient
    "sensitivity_pct": 0.20,
    "mc_iterations": 10_000,
    "bootstrap_iterations": 10_000,
    "dsr_alpha": 0.05,
    "t_stat_threshold": 2.0,
    "t_stat_multiple": 3.0,
    "wfe_threshold": 0.50,
    "cdar_beta": 0.90,
    "dynamic_caps": [0.10, 0.15, 0.20, 0.25, 0.33],
    "max_position_pct": 0.33,
    "kelly_fraction": 0.50,
    "benchmark_tickers": ["SPY", "ACWI"],
    "random_state": 42,
}
