"""Canonical CLI parser for no-rebalance replay scripts.

Extracted from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` lines
~1798-1842 (2026-04-30). Strategy-agnostic.

Usage in a strategy adapter:

    from strategy_tester.s5_replay import cli

    parser = cli.build_parser(description="MyStrategy S5 no-rebalance replay")
    parser.add_argument("--my-strategy-flag", ...)  # adapter extras
    args = parser.parse_args()
"""
from __future__ import annotations

import argparse

from strategy_tester.s5_replay.walker import (
    CASH_BUFFER,
    DEFAULT_DD_TOL,
    DEFAULT_PER_NAME_CAP,
    DEFAULT_SEED_NAV,
    DEFAULT_SIZING_RULE,
    SIZING_RULES,
)


def build_parser(
    description: str = "S5 no-rebalance replay (24 schemes)",
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed-nav", type=float, default=DEFAULT_SEED_NAV)
    p.add_argument("--cap", type=str, default="all",
                   help="Filter to cap value (5/10/15/20) or 'all'")
    p.add_argument("--schemes", type=str, default=None,
                   help="Comma-separated subset of schemes")
    p.add_argument("--cash-buffer", type=float, default=CASH_BUFFER)
    p.add_argument(
        "--sizing-rule", type=str, default=DEFAULT_SIZING_RULE,
        choices=list(SIZING_RULES),
        help=(
            "netliq_clip: target=w*NetLiq, clip at cash*buffer (legacy). "
            "cash_fraction: target=w*cash*buffer, no clipping. "
            "paleologo_strict: target=cash*buffer*min(w,cap) [+ DD throttle]."
        ),
    )
    p.add_argument(
        "--dd-throttle", type=str, default="off",
        choices=("off", "linear"),
        help=(
            "Paleologo APM Ch.10 DD throttle: scale entry size by "
            "max(0, 1 - DD/dd_tol). Only active for sizing-rule="
            "paleologo_strict."
        ),
    )
    p.add_argument(
        "--dd-tol", type=float, default=DEFAULT_DD_TOL,
        help=(
            f"Drawdown tolerance for the throttle. Default: "
            f"{DEFAULT_DD_TOL}."
        ),
    )
    p.add_argument(
        "--per-name-cap", type=float, default=DEFAULT_PER_NAME_CAP,
        help=(
            f"Default single-name cap for static schemes (no cap suffix). "
            f"Schemes with cap suffix (e.g. `*_cap10`) use the suffix "
            f"value. Default: {DEFAULT_PER_NAME_CAP}."
        ),
    )
    return p


def filter_schemes(
    all_names: list[str], cap: str | None, schemes: str | None,
) -> list[str]:
    """Filter scheme list by cap value or explicit name list."""
    import logging
    log = logging.getLogger(__name__)
    keep = list(all_names)
    if schemes:
        wanted = {s.strip() for s in schemes.split(",") if s.strip()}
        keep = [s for s in keep if s in wanted]
        missing = wanted - set(keep)
        if missing:
            log.warning("  Unknown schemes ignored: %s", sorted(missing))
    if cap and cap.lower() != "all":
        cap_int = int(cap)
        suffix = f"cap{cap_int:02d}"
        keep = [
            s for s in keep
            if s in {"equal_weight", "sharpe_weighted", "inverse_vol", "hrp"}
            or s.endswith(suffix)
        ]
    return keep
