"""Shared S5 no-rebalance replay infrastructure.

Strategy-agnostic library extracted from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` (2026-04-30).
Each strategy provides a thin adapter that:

1. Builds a per-ticker ``cache`` matching the schema in ``walker.walk_portfolio_oracle``
2. Calls ``runner.run_all_schemes`` (or ``run_scheme`` per scheme)
3. Hands the results to ``report.build_html``

Modules:
    walker  — dataclasses (TickerState, TradeLog, FailedEntry, ReplayState) +
              ``walk_portfolio_oracle`` (the cash-aware day-by-day simulator)
    oracles — static base maps + scheme name parser + ``build_oracle`` factory
    metrics — ``equity_metrics``, ``vs_spy``, ``summary_rows``, ``spy_summary_row``
    runner  — ``run_scheme``, ``run_all_schemes`` + ``SchemeResult`` dataclass
    report  — ``build_html`` + equity/drawdown/cash-mobilization overlays
    cli     — ``build_parser`` (the canonical CLI flags)

See ``docs/PORTFOLIO_WEIGHTING_METHODS.md`` for the canonical
4-layer model (rank × cap × cash × throttle).
"""
from __future__ import annotations

from strategy_tester.s5_replay.walker import (
    ANNUALIZE,
    CASH_BUFFER,
    COST_PER_SIDE,
    DEFAULT_DD_TOL,
    DEFAULT_PER_NAME_CAP,
    DEFAULT_SEED_NAV,
    DEFAULT_SIZING_RULE,
    SIZING_RULES,
    SLIP_BPS,
    FailedEntry,
    ReplayState,
    TickerState,
    TradeLog,
    walk_portfolio_oracle,
)

__all__ = [
    "ANNUALIZE",
    "CASH_BUFFER",
    "COST_PER_SIDE",
    "DEFAULT_DD_TOL",
    "DEFAULT_PER_NAME_CAP",
    "DEFAULT_SEED_NAV",
    "DEFAULT_SIZING_RULE",
    "SIZING_RULES",
    "SLIP_BPS",
    "FailedEntry",
    "ReplayState",
    "TickerState",
    "TradeLog",
    "walk_portfolio_oracle",
]
