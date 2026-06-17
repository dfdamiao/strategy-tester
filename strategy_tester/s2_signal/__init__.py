"""S2 signal methods — auto-register on import."""
from strategy_tester.s2_signal import zscore_robust_mad  # noqa: F401
from strategy_tester.s2_signal import zscore_standard  # noqa: F401
from strategy_tester.s2_signal import kalman_hedge  # noqa: F401
from strategy_tester.s2_signal import bollinger  # noqa: F401
from strategy_tester.s2_signal import ma_crossover  # noqa: F401
from strategy_tester.s2_signal import ema_crossover  # noqa: F401
from strategy_tester.s2_signal import dual_ma_crossover  # noqa: F401
from strategy_tester.s2_signal import kama_crossover  # noqa: F401
from strategy_tester.s2_signal import momentum  # noqa: F401
# --- Alternative strategies (Phase 1) ---
from strategy_tester.s2_signal import rsi_simple  # noqa: F401
from strategy_tester.s2_signal import rsi_wilder  # noqa: F401
from strategy_tester.s2_signal import vol_adaptive_zscore  # noqa: F401
from strategy_tester.s2_signal import adx_regime_gate  # noqa: F401
# --- Alternative strategies (Phase 2) ---
from strategy_tester.s2_signal import donchian  # noqa: F401
from strategy_tester.s2_signal import atr_pullback  # noqa: F401
from strategy_tester.s2_signal import vol_forecast_gate  # noqa: F401
# --- Alternative strategies (Phase 3) ---
from strategy_tester.s2_signal import halflife_time_decay  # noqa: F401
from strategy_tester.s2_signal import cointegration_spread  # noqa: F401
from strategy_tester.s2_signal import regime_switch  # noqa: F401
# --- Ratio breakout ---
from strategy_tester.s2_signal import trendline_breakout  # noqa: F401
# --- Quantitativo Tier 1 (book-canonical fillers, 2026-04-27) ---
from strategy_tester.s2_signal import cum_rsi  # noqa: F401
from strategy_tester.s2_signal import adx_ema_pullback  # noqa: F401
# --- Connors RSI(2) single-asset MR (A1 canonical, 2026-05-09) ---
from strategy_tester.s2_signal import rsi_connors  # noqa: F401
