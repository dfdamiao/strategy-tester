"""Pattern detectors (trendlines, pivots, etc.) used by signal modules."""
from strategy_tester.detectors.trendline import (
    HybridTrendDetector,
    TrendlineResult,
    detect_resistance,
    detect_support,
)

__all__ = [
    "HybridTrendDetector",
    "TrendlineResult",
    "detect_resistance",
    "detect_support",
]
