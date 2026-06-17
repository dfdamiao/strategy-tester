"""Cash-aware day-by-day simulator + dataclasses.

Extracted verbatim from
``obv_pivot/portfolio_analysis/scripts/cash_aware_replay.py`` (dataclasses +
constants) and
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py``
(``walk_portfolio_oracle``).

Strategy-agnostic: the walker reads a ``cache: dict[str, dict]`` whose
schema is contract-fixed (see ``walk_portfolio_oracle`` docstring). All
strategy-specific things (signal, ATR, stop) live in the cache itself; the
walker only calls the supplied ``weight_oracle`` to obtain per-bar weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (from cash_aware_replay.py)
# ---------------------------------------------------------------------------

ANNUALIZE = 252
DEFAULT_SEED_NAV = 100_000.0
CASH_BUFFER = 0.95          # cap entries at 95% of cash projection
COST_PER_SIDE = 0.001       # 10 bps/side (Chan AT 2013 Ch.3) — LEGACY
SLIP_BPS = 5.0 / 1e4        # 5 bps slippage

# ---------------------------------------------------------------------------
# IBKR per-share commission model (added 2026-05-18 per PORTFOLIO_CONSTRUCTION.md §5)
# ---------------------------------------------------------------------------
# Replaces the legacy 10 bps notional model. IBKR Pro Fixed schedule for
# US stocks/ETFs (verified 2026-05-18: $1 minimum confirmed live in
# ibkr_v2/tests/integration/level7_round_trip.py — 1-share AAPL test).
#
# Models supported:
#   "flat_10bps"      — legacy Chan AT 2013, kept for back-compat with
#                       the 9 deployable v2 strategies (2026-06-01 cohort)
#   "ibkr_pro_fixed"  — $0.005/share, min $1.00/order, max 1% of trade value
#                       (DEFAULT for new strategies as of 2026-05-18)
#   "ibkr_pro_tiered" — $0.0035/share base tier (≤300K sh/mo), min $0.35,
#                       plus ~$0.0008/share pass-through (SEC + FINRA +
#                       clearing) — approximation; live tiered rates
#                       depend on volume tier
#   "ibkr_lite"       — $0 commission on US stocks/ETFs

COMMISSION_MODELS = (
    "flat_10bps", "ibkr_pro_fixed", "ibkr_pro_tiered", "ibkr_lite",
)
# Restored to `ibkr_pro_fixed` 2026-05-18 PM after the ratio-aware fix
# (per PORTFOLIO_CONSTRUCTION.md §5.1). Ratios go through 2× per-leg
# commission via `is_ratio=True` rather than `shares × per_share` on the
# synthetic "share" count, which used to balloon to 40K+ for sub-$1
# synthetic prices and hit the 1% cap. See `commission_per_side` below.
DEFAULT_COMMISSION_MODEL = "ibkr_pro_fixed"

# IBKR Pro Fixed
IBKR_FIXED_PER_SHARE = 0.005
IBKR_FIXED_MIN_ORDER = 1.00
IBKR_FIXED_MAX_PCT   = 0.01     # 1% of trade value cap

# IBKR Pro Tiered (base tier)
IBKR_TIERED_PER_SHARE  = 0.0035
IBKR_TIERED_MIN_ORDER  = 0.35
IBKR_TIERED_PASSTHROUGH_PER_SHARE = 0.0008  # SEC + FINRA + clearing approx
IBKR_TIERED_MAX_PCT    = 0.01

# Ratio-aware approximation: a synthetic ratio (e.g. SPY/QQQ traded as one
# unit) is executed live as ONE leg — long the numerator only — for the
# strategies in this repo (signal generated from num/den ratio, but the
# trade is just the numerator). The synthetic "shares" count is meaningless
# (often 40K+ when synth price < $1), so for per-share commission models
# we back out the real leg share count from `notional / leg_price` where
# `leg_price` is the numerator's per-bar close (passed in via the walker
# cache). When `leg_price` is unavailable (synthetic test caches without
# strategy plumbing), we fall back to AVG_ETF_PRICE = $50, the midpoint
# of the typical US ETF price band ($20–$500). The fallback over- or
# under-counts at the per-trade level but is dominated by the $1 / $0.35
# minimum at all reasonable ETF prices anyway. See
# PORTFOLIO_CONSTRUCTION.md §5.3 for the share-price sensitivity.
AVG_ETF_PRICE = 50.0


def commission_per_side(
    shares: int,
    notional: float,
    *,
    model: str = DEFAULT_COMMISSION_MODEL,
    flat_tx_cost: float = COST_PER_SIDE,
    is_ratio: bool = False,
    leg_price: float = 0.0,
) -> float:
    """Compute IBKR commission for ONE side (entry or exit) of a trade.

    Parameters
    ----------
    shares : int
        Order size in shares (absolute value; sign-independent). For
        synthetic ratio instruments this is the synthetic share count
        (often inflated when synth price < $1); ignored under
        ``is_ratio=True`` for per-share models — the real leg share
        count is derived from ``notional / leg_price`` instead.
    notional : float
        Trade value = ``shares × price``. For ratios this is the dollar
        notional of the single numerator leg.
    model : str
        One of ``COMMISSION_MODELS``. ``"flat_10bps"`` reproduces the
        legacy ``notional × COST_PER_SIDE`` formula for back-compat.
    flat_tx_cost : float
        Used only when ``model="flat_10bps"``. Default ``COST_PER_SIDE``.
    is_ratio : bool
        If True, treat this trade as a synthetic ratio executed as a
        SINGLE numerator leg (the strategies in this repo are long-only-
        numerator on ratios). Only affects ``ibkr_pro_fixed`` and
        ``ibkr_pro_tiered``: leg share count is derived from ``notional
        / leg_price`` (fallback ``AVG_ETF_PRICE`` when leg_price ≤ 0).
        ``flat_10bps`` and ``ibkr_lite`` are unaffected.
    leg_price : float
        Numerator's per-bar close (for ratios). Used only when
        ``is_ratio=True``. Default 0.0 triggers ``AVG_ETF_PRICE``
        fallback.

    Returns
    -------
    float : commission in account currency (USD for US stocks/ETFs).
    """
    if shares <= 0 or notional <= 0:
        return 0.0
    if model == "flat_10bps":
        return notional * flat_tx_cost
    if model == "ibkr_lite":
        return 0.0
    # Per-share models: effective share count differs for ratios (1-leg,
    # actual numerator share count) vs singles (walker's `shares`).
    if is_ratio:
        px = leg_price if leg_price > 0 else AVG_ETF_PRICE
        eff_shares = notional / px
    else:
        eff_shares = shares
    if model == "ibkr_pro_fixed":
        base = max(IBKR_FIXED_MIN_ORDER, eff_shares * IBKR_FIXED_PER_SHARE)
        return min(base, notional * IBKR_FIXED_MAX_PCT)
    if model == "ibkr_pro_tiered":
        base = max(IBKR_TIERED_MIN_ORDER, eff_shares * IBKR_TIERED_PER_SHARE)
        passthrough = eff_shares * IBKR_TIERED_PASSTHROUGH_PER_SHARE
        return min(base + passthrough, notional * IBKR_TIERED_MAX_PCT)
    raise ValueError(
        f"commission model {model!r} not in {COMMISSION_MODELS}"
    )

# Sizing rules ---------------------------------------------------------------
# netliq_clip              — Option B (legacy): target = w × NetLiq, clipped
#                            at cash × buffer. Produces "clipped" failed
#                            entries when fully invested. Audit comparison
#                            only.
# cash_fraction            — Option B': snapshot cash_pool = cash × buffer
#                            ONCE before entry loop, target = w × cash_pool.
#                            No cap → per-position can hit 95% of cash when
#                            oracle returns w=1.0 (n_active=1 case).
# cash_fraction_capped     — Snapshot-cap variant (2026-05-17): bar-start
#                            snapshot + per-name cap. target = min(w, cap)
#                            × cash × buffer. Residual stays uncalled. All
#                            entries on a bar are sized off the SAME
#                            snapshot — order-invariant, but multiple
#                            simultaneous entries can collectively exceed
#                            free cash and trigger the post-loop reduce-by-
#                            one clamp. No DD throttle.
# cash_fraction_seq_capped — NEW (2026-05-18, layer-0 invariant): sequential
#                            cash consumption. target = min(w, cap) ×
#                            CURRENT state.cash × buffer, re-read INSIDE the
#                            entry loop. Each successful entry deducts cash
#                            from the pool so the next entry sees the post-
#                            deduction balance. Matches Daniel's "50% then
#                            40% of remaining" intent (2026-05-17 doc
#                            comment c0). Strictly conservative vs the
#                            snapshot variant — never collectively exceeds
#                            free cash by construction.
# paleologo_strict         — Snapshot cap + ALWAYS-ON linear DD throttle +
#                            cash discipline. Same snapshot mechanic as
#                            cash_fraction_capped PLUS automatic linear
#                            throttle: target *= max(0, 1 − DD/DD_tol). The
#                            throttle is what distinguishes paleologo_strict
#                            from cash_fraction_capped; without it the two
#                            rules are identical (ref-doc Q13 c40). As of
#                            2026-05-18 the `dd_throttle` arg is IGNORED
#                            when sizing_rule="paleologo_strict" — throttle
#                            is always linear. This is the LINEARIZED
#                            Grossman-Zhou (1993) form endorsed by
#                            Paleologo APM (2021) Ch.9 fn.4 + §11.4 (book
#                            p.151 fn.1 endorses the linear approximation
#                            as the standard practitioner implementation).
#                            The verbatim Paleologo form is f = S₀·(1 −
#                            DD_tol/DD) (inverted ratio); the /k(t)
#                            bucket-sizing divisor often co-cited with
#                            Paleologo is in-house — not in the book.
#
# No-touch invariant (Layer-0, codified 2026-05-18 per ref-doc c0/c4/c52):
# the walker NEVER trims, downsizes, or closes incumbents to make room for
# new entries. Step 1 exits run only on state-machine signals (pos_raw=0)
# or stop-loss triggers. Step 2 fresh entries are sized exclusively from
# free cash (snapshot or sequential, per the sizing rule above). Re-entry
# while in_position is a no-op (one position per name until exit). DD
# throttle, when ON, scales only NEW-ENTRY sizing — never incumbents.
SIZING_RULES = (
    "netliq_clip", "cash_fraction", "cash_fraction_capped",
    "cash_fraction_seq_capped", "paleologo_strict",
)
DEFAULT_SIZING_RULE = "cash_fraction_seq_capped"  # bumped 2026-05-18 (Layer-0)
DEFAULT_DD_TOL = 0.25       # canonical default (per-strategy may override)
DEFAULT_PER_NAME_CAP = 0.20  # bumped 2026-05-18 from 0.10 — fits the wider
# canonical cap sweep {5,10,20,30,50}% (ref-doc Q4 / Q12, 2026-05-17 c39).




# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TickerState:
    """Live position state for one ticker."""
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    peak_close: float
    trail_level: float  # for ATR trail stops
    stop_code: int
    stop_param: float
    base_weight: float
    # Cap-verification diagnostics captured at entry (paleologo_strict)
    entry_pct_cash: float = 0.0   # cost / cash_at_bar_start × 100
    entry_pct_nav: float = 0.0    # cost / NAV_at_entry × 100
    # Per-bar excursion tracking (MAE = worst unrealized return %,
    # MFE = best unrealized return %). Both initialised at 0% because
    # entry-bar close = entry_price (no excursion yet). Updated on each bar
    # the position is held, including the exit bar before close-out.
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    # Entry-bar numerator price for ratios (1-leg commission back-out).
    # 0.0 when not provided in cache → commission_per_side falls back to
    # AVG_ETF_PRICE. For singles this stays 0.0 (per-share commission
    # uses walker `shares` directly).
    entry_leg_price: float = 0.0


@dataclass
class TradeLog:
    """One closed trade record."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str  # "signal" | "stop_atr" | "stop_pct" | "force_eow"
    # Diagnostics for cap verification (paleologo_strict)
    entry_pct_cash: float = 0.0   # position_$ / cash_at_bar_start (entry)
    entry_pct_nav: float = 0.0    # position_$ / NAV_at_entry
    # Max adverse / favorable excursion during the holding period (in %)
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    hold_days: int = 0            # (exit_date - entry_date).days
    # Total commission paid on this round trip (entry + exit)
    # = shares × (entry_price + exit_price) × tx_cost
    commission_dollars: float = 0.0


@dataclass
class FailedEntry:
    """One signal that failed to enter due to cash constraints."""
    ticker: str
    date: pd.Timestamp
    target_dollars: float
    sized_dollars: float  # 0 = skipped, < target = clipped
    cash_available: float
    reason: str  # "no_cash" | "clipped" | "too_small"


@dataclass
class ReplayState:
    """Mutable state through the walk."""
    cash: float
    positions: dict[str, TickerState] = field(default_factory=dict)
    daily_snapshot: list[dict] = field(default_factory=list)
    trades: list[TradeLog] = field(default_factory=list)
    failed_entries: list[FailedEntry] = field(default_factory=list)
    # MTM peak for DD throttle (paleologo_strict only); refreshes upward
    peak_nav: float = 0.0
    # Cap-binding telemetry (added 2026-05-18 per ref-doc Q17 c44):
    # how often the per-name cap actually bound at entry time.
    n_cap_bound: int = 0      # entries where min(w, cap) == cap
    n_entries_total: int = 0  # all attempted entries (cap-bound or not)


# ---------------------------------------------------------------------------
# Generalised cash-aware walk
# ---------------------------------------------------------------------------


def walk_portfolio_oracle(
    cache: dict[str, dict],
    weight_oracle: Callable[[set[str], pd.Timestamp], dict[str, float]],
    seed_nav: float,
    buffer: float = CASH_BUFFER,
    tx_cost: float = COST_PER_SIDE,
    slip: float = SLIP_BPS,
    sizing_rule: str = DEFAULT_SIZING_RULE,
    per_name_cap: float = DEFAULT_PER_NAME_CAP,
    dd_tol: float = DEFAULT_DD_TOL,
    dd_throttle: str = "off",
    commission_model: str = DEFAULT_COMMISSION_MODEL,
) -> ReplayState:
    """Per-entry weight from ``weight_oracle(active_set, date)``.

    Cache schema (per ticker):
        index       — pd.DatetimeIndex of trading days
        close       — np.ndarray[float64]
        low         — np.ndarray[float64]      (for stop-pct trigger)
        atr         — np.ndarray[float64]      (Wilder ATR for stop-atr trail)
        pos_raw     — np.ndarray[int8]         (0=flat, 1=long; signal output)
        stop_code   — int (0=none, 1=stop-pct, 2=stop-atr)
        stop_param  — float (pct or k for ATR multiplier)
        base_weight — float (per-ticker static weight, can be 1.0)
        is_ratio    — bool (OPTIONAL, default False; 2026-05-18 commission
                      audit). True for synthetic ratio instruments (e.g.
                      ETF_A/ETF_B traded as one unit). Routes per-share
                      commission models through the single-leg numerator-
                      only formula so they don't balloon on inflated
                      synthetic share counts. No effect on ``flat_10bps``
                      / ``ibkr_lite``.
        leg_price   — np.ndarray[float64] (OPTIONAL, per-bar; 2026-05-18).
                      Real-world per-share price used to back out leg
                      shares when ``is_ratio=True``: numerator's close
                      for ratios. Same length as ``close``. Falls back
                      to ``AVG_ETF_PRICE`` when absent. Ignored for
                      singles and for ``flat_10bps`` / ``ibkr_lite``.

    Five sizing rules (see SIZING_RULES module docstring for full table):

    * ``netliq_clip`` — legacy Option B
    * ``cash_fraction`` — uncapped snapshot
    * ``cash_fraction_capped`` — snapshot + per-name cap
    * ``cash_fraction_seq_capped`` — sequential cash consumption + cap
      (DEFAULT 2026-05-18, layer-0 invariant)
    * ``paleologo_strict`` — snapshot cap + optional DD throttle
      (Paleologo APM Ch.10 dynamic-1/k). DD is MTM-based (peak NAV
      refreshes upward; current NAV = cash + Σ MTM positions).
    """
    if sizing_rule not in SIZING_RULES:
        raise ValueError(
            f"sizing_rule={sizing_rule!r} not in {SIZING_RULES}"
        )
    if dd_throttle not in ("off", "linear"):
        raise ValueError(
            f"dd_throttle={dd_throttle!r} must be 'off' or 'linear'"
        )
    state = ReplayState(cash=seed_nav)
    state.peak_nav = seed_nav  # MTM peak; refreshes upward only
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(set(c["index"]) for c in cache.values())))
    )
    ticker_idx: dict[str, dict[pd.Timestamp, int]] = {
        t: {d: i for i, d in enumerate(c["index"])} for t, c in cache.items()
    }

    for date in all_dates:
        # Step 1 — exits / stops
        exits_today: list[str] = []
        for ticker in list(state.positions.keys()):
            pos = state.positions[ticker]
            c = cache[ticker]
            ti = ticker_idx[ticker].get(date)
            if ti is None:
                continue
            close_t = c["close"][ti]
            low_t = c["low"][ti]
            atr_t = c["atr"][ti]
            pos_raw_t = int(c["pos_raw"][ti])

            # Update MAE/MFE based on this bar's close BEFORE any exit logic
            # so exit-day excursion is captured even on signal exits.
            unrealized = (close_t / pos.entry_price - 1.0) * 100.0
            if unrealized < pos.mae_pct:
                pos.mae_pct = unrealized
            if unrealized > pos.mfe_pct:
                pos.mfe_pct = unrealized

            stop_hit = False
            stop_px = 0.0
            if pos.stop_code == 1:
                floor = pos.entry_price * (1.0 - pos.stop_param)
                if low_t <= floor:
                    stop_hit = True
                    stop_px = floor * (1.0 - slip)
            elif pos.stop_code == 2:
                cand = close_t - pos.stop_param * atr_t
                if cand > pos.trail_level:
                    pos.trail_level = cand
                if close_t <= pos.trail_level:
                    stop_hit = True
                    stop_px = pos.trail_level * (1.0 - slip)

            if stop_hit:
                stop_pct = (stop_px / pos.entry_price - 1.0) * 100.0
                mae_final = min(pos.mae_pct, stop_pct)
                mfe_final = max(pos.mfe_pct, stop_pct)
                # Per-side commission (model-aware): entry side was charged
                # below at entry time; only the exit side is new here. The
                # legacy `flat_10bps` model lumped both sides into one
                # commission_dollars field — preserve that behaviour by
                # summing entry+exit commissions on the exit-side log.
                is_ratio_t = bool(c.get("is_ratio", False))
                leg_price_arr = c.get("leg_price")
                exit_leg_price = (
                    float(leg_price_arr[ti]) if leg_price_arr is not None else 0.0
                )
                entry_notional = pos.shares * pos.entry_price
                exit_notional = pos.shares * stop_px
                entry_comm = commission_per_side(
                    pos.shares, entry_notional,
                    model=commission_model, flat_tx_cost=tx_cost,
                    is_ratio=is_ratio_t, leg_price=pos.entry_leg_price,
                )
                exit_comm = commission_per_side(
                    pos.shares, exit_notional,
                    model=commission_model, flat_tx_cost=tx_cost,
                    is_ratio=is_ratio_t, leg_price=exit_leg_price,
                )
                commission = entry_comm + exit_comm
                proceeds = pos.shares * stop_px - exit_comm
                state.cash += proceeds
                state.trades.append(TradeLog(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, exit_date=date,
                    exit_price=stop_px, shares=pos.shares,
                    pnl_dollars=proceeds - pos.shares * pos.entry_price,
                    pnl_pct=(stop_px / pos.entry_price - 1.0) * 100,
                    exit_reason="stop_atr" if pos.stop_code == 2 else "stop_pct",
                    entry_pct_cash=pos.entry_pct_cash,
                    entry_pct_nav=pos.entry_pct_nav,
                    mae_pct=mae_final, mfe_pct=mfe_final,
                    hold_days=(date - pos.entry_date).days,
                    commission_dollars=commission,
                ))
                exits_today.append(ticker)
            elif pos_raw_t == 0:
                exit_px = close_t * (1.0 - slip)
                exit_pct = (exit_px / pos.entry_price - 1.0) * 100.0
                mae_final = min(pos.mae_pct, exit_pct)
                mfe_final = max(pos.mfe_pct, exit_pct)
                is_ratio_t = bool(c.get("is_ratio", False))
                leg_price_arr = c.get("leg_price")
                exit_leg_price = (
                    float(leg_price_arr[ti]) if leg_price_arr is not None else 0.0
                )
                entry_notional = pos.shares * pos.entry_price
                exit_notional = pos.shares * exit_px
                entry_comm = commission_per_side(
                    pos.shares, entry_notional,
                    model=commission_model, flat_tx_cost=tx_cost,
                    is_ratio=is_ratio_t, leg_price=pos.entry_leg_price,
                )
                exit_comm = commission_per_side(
                    pos.shares, exit_notional,
                    model=commission_model, flat_tx_cost=tx_cost,
                    is_ratio=is_ratio_t, leg_price=exit_leg_price,
                )
                commission = entry_comm + exit_comm
                proceeds = pos.shares * exit_px - exit_comm
                state.cash += proceeds
                state.trades.append(TradeLog(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, exit_date=date,
                    exit_price=exit_px, shares=pos.shares,
                    pnl_dollars=proceeds - pos.shares * pos.entry_price,
                    pnl_pct=(exit_px / pos.entry_price - 1.0) * 100,
                    exit_reason="signal",
                    entry_pct_cash=pos.entry_pct_cash,
                    entry_pct_nav=pos.entry_pct_nav,
                    mae_pct=mae_final, mfe_pct=mfe_final,
                    hold_days=(date - pos.entry_date).days,
                    commission_dollars=commission,
                ))
                exits_today.append(ticker)
            else:
                if close_t > pos.peak_close:
                    pos.peak_close = close_t

        for t in exits_today:
            state.positions.pop(t, None)

        # Step 2 — fresh ENTER signals
        entry_candidates: dict[str, dict] = {}
        for ticker, c in cache.items():
            if ticker in state.positions:
                continue
            ti = ticker_idx[ticker].get(date)
            if ti is None or ti < 1:
                continue
            pos_now = int(c["pos_raw"][ti])
            pos_prev = int(c["pos_raw"][ti - 1])
            if pos_now == 1 and pos_prev == 0:
                leg_price_arr = c.get("leg_price")
                entry_candidates[ticker] = {
                    "close": c["close"][ti],
                    "atr": c["atr"][ti],
                    "stop_code": c["stop_code"],
                    "stop_param": c["stop_param"],
                    "base_weight": c["base_weight"],
                    "leg_price": (
                        float(leg_price_arr[ti])
                        if leg_price_arr is not None else 0.0
                    ),
                }

        if entry_candidates:
            active_set = set(state.positions) | set(entry_candidates)
            oracle_w = weight_oracle(active_set, date)
            entry_weights = {t: oracle_w.get(t, 0.0) for t in entry_candidates}

            netliq = state.cash + sum(
                state.positions[t].shares * cache[t]["close"][ticker_idx[t][date]]
                for t in state.positions
                if ticker_idx[t].get(date) is not None
            )
            # MTM-NAV peak refresh + current DD (used by paleologo_strict
            # DD throttle only; computed here once per bar).
            if netliq > state.peak_nav:
                state.peak_nav = netliq
            dd_now = (
                (state.peak_nav - netliq) / state.peak_nav
                if state.peak_nav > 0 else 0.0
            )
            # paleologo_strict: linear DD throttle is now ALWAYS ON
            # (2026-05-18, ref-doc). The `dd_throttle` arg is retained for
            # API back-compat but ignored when sizing_rule="paleologo_strict"
            # — the rule is what distinguishes it from cash_fraction_capped,
            # so the throttle is load-bearing for the rule's identity.
            throttle_factor = (
                max(0.0, 1.0 - dd_now / dd_tol)
                if (sizing_rule == "paleologo_strict" and dd_tol > 0)
                else 1.0
            )
            # Snapshot cash bucket BEFORE the loop so cash_fraction /
            # paleologo_strict sizing is invariant to per-entry submission
            # order (otherwise we re-introduce first-mover cash-depletion
            # bias).
            cash_pool_t = max(0.0, state.cash * buffer)

            sorted_entries = sorted(
                entry_candidates.items(),
                key=lambda kv: entry_weights.get(kv[0], 0.0),
                reverse=True,
            )

            for ticker, info in sorted_entries:
                w = entry_weights.get(ticker, 0.0)
                if w <= 0:
                    continue
                state.n_entries_total += 1
                cap_bound_here = False
                if sizing_rule == "paleologo_strict":
                    # Hard per-position cap, on cash, optional DD throttle.
                    # target_$ = cash × buffer × min(w, cap) × throttle
                    capped_w = min(w, per_name_cap)
                    cap_bound_here = capped_w < w - 1e-12
                    target_dollars = capped_w * cash_pool_t * throttle_factor
                    sized_dollars = target_dollars
                elif sizing_rule == "cash_fraction_capped":
                    # Snapshot-cap variant (2026-05-17).
                    # target_$ = min(w, cap) × bar-start-cash × buffer.
                    # All entries on a bar size off the SAME snapshot →
                    # order-invariant, can collectively exceed free cash
                    # (post-loop reduce-by-one then clamps to actual cash).
                    capped_w = min(w, per_name_cap)
                    cap_bound_here = capped_w < w - 1e-12
                    target_dollars = capped_w * cash_pool_t
                    sized_dollars = target_dollars
                elif sizing_rule == "cash_fraction_seq_capped":
                    # Sequential-cash-consumption variant (Layer-0 invariant,
                    # 2026-05-18). target_$ = min(w, cap) × CURRENT cash ×
                    # buffer, re-read inside the loop. Strictly conservative
                    # vs snapshot — never collectively exceeds free cash
                    # because each successful entry's `state.cash -= cost`
                    # below shrinks the next entry's pool. Matches Daniel's
                    # "50% then 40% of remaining" intent (ref-doc c0).
                    capped_w = min(w, per_name_cap)
                    cap_bound_here = capped_w < w - 1e-12
                    free_now = max(0.0, state.cash * buffer)
                    target_dollars = capped_w * free_now
                    sized_dollars = target_dollars
                elif sizing_rule == "cash_fraction":
                    target_dollars = w * cash_pool_t
                    sized_dollars = target_dollars
                else:  # netliq_clip
                    target_dollars = w * netliq
                    headroom = max(0.0, state.cash * buffer)
                    sized_dollars = min(target_dollars, headroom)
                if cap_bound_here:
                    state.n_cap_bound += 1

                entry_px = info["close"] * (1.0 + slip)
                # Sizing heuristic: use the legacy `cost_per_share` (price
                # × (1 + 10 bps)) as a conservative initial share count.
                # Then compute the ACTUAL cash cost using the model-aware
                # commission and back off if it overshoots state.cash. This
                # ensures we never deploy more cash than `sized_dollars`,
                # regardless of which commission model is active.
                cost_per_share_naive = entry_px * (1.0 + tx_cost)
                shares = (
                    int(sized_dollars / cost_per_share_naive)
                    if cost_per_share_naive > 0 else 0
                )
                # paleologo_strict + cash_fraction + cash_fraction_capped
                # never run out of cash by construction (Σ targets ≤
                # cash × buffer); netliq_clip can.
                no_clip_modes = {
                    "cash_fraction", "cash_fraction_capped",
                    "cash_fraction_seq_capped", "paleologo_strict",
                }
                if shares <= 0:
                    reason = (
                        "too_small" if sizing_rule in no_clip_modes
                        else "no_cash"
                    )
                    state.failed_entries.append(FailedEntry(
                        ticker=ticker, date=date,
                        target_dollars=target_dollars, sized_dollars=0.0,
                        cash_available=state.cash, reason=reason,
                    ))
                    continue
                # Model-aware actual cost (commission added to notional)
                is_ratio_t = bool(cache[ticker].get("is_ratio", False))
                entry_leg_price = float(info.get("leg_price", 0.0))
                entry_notional = shares * entry_px
                entry_comm = commission_per_side(
                    shares, entry_notional,
                    model=commission_model, flat_tx_cost=tx_cost,
                    is_ratio=is_ratio_t, leg_price=entry_leg_price,
                )
                cost = entry_notional + entry_comm
                if cost > state.cash:
                    shares -= 1
                    if shares > 0:
                        entry_notional = shares * entry_px
                        entry_comm = commission_per_side(
                            shares, entry_notional,
                            model=commission_model, flat_tx_cost=tx_cost,
                            is_ratio=is_ratio_t, leg_price=entry_leg_price,
                        )
                        cost = entry_notional + entry_comm
                    if shares <= 0:
                        reason = (
                            "too_small" if sizing_rule in no_clip_modes
                            else "no_cash"
                        )
                        state.failed_entries.append(FailedEntry(
                            ticker=ticker, date=date,
                            target_dollars=target_dollars, sized_dollars=0.0,
                            cash_available=state.cash, reason=reason,
                        ))
                        continue

                state.cash -= cost
                if sizing_rule == "netliq_clip":
                    clipped = sized_dollars < target_dollars * 0.99
                    if clipped:
                        state.failed_entries.append(FailedEntry(
                            ticker=ticker, date=date,
                            target_dollars=target_dollars, sized_dollars=cost,
                            cash_available=state.cash + cost,
                            reason="clipped",
                        ))

                trail_init = (
                    info["close"] - info["stop_param"] * info["atr"]
                    if info["stop_code"] == 2 else 0.0
                )
                # Capture cap-verification diagnostics. cash_pool_t was
                # snapshot before this loop (= cash_at_bar_start × buffer).
                cash_at_bar_start = (
                    cash_pool_t / buffer if buffer > 0 else 0.0
                )
                entry_pct_cash = (
                    cost / cash_at_bar_start * 100.0
                    if cash_at_bar_start > 0 else 0.0
                )
                entry_pct_nav = (
                    cost / netliq * 100.0 if netliq > 0 else 0.0
                )
                state.positions[ticker] = TickerState(
                    ticker=ticker, shares=shares,
                    entry_price=entry_px, entry_date=date,
                    peak_close=info["close"], trail_level=trail_init,
                    stop_code=info["stop_code"], stop_param=info["stop_param"],
                    base_weight=info["base_weight"],
                    entry_pct_cash=entry_pct_cash,
                    entry_pct_nav=entry_pct_nav,
                    entry_leg_price=entry_leg_price,
                )

        # Step 3 — EOD snapshot. Refresh peak NAV upward and record DD.
        position_value = 0.0
        per_pos_mtm: dict[str, float] = {}
        for t, p in state.positions.items():
            ti = ticker_idx[t].get(date)
            if ti is not None:
                mtm = p.shares * cache[t]["close"][ti]
            else:
                mtm = p.shares * p.entry_price
            per_pos_mtm[t] = float(mtm)
            position_value += mtm
        netliq = state.cash + position_value
        if netliq > state.peak_nav:
            state.peak_nav = netliq
        dd_eod = (
            (state.peak_nav - netliq) / state.peak_nav
            if state.peak_nav > 0 else 0.0
        )
        # Per-bar weight vector — fraction of NAV in each position (excludes
        # cash). Used downstream (runner.py) for HHI / Meucci ENB
        # concentration diagnostics (ref-doc Q5 / Q21, 2026-05-17).
        weights = (
            {t: v / netliq for t, v in per_pos_mtm.items()}
            if netliq > 1e-9 else {}
        )
        state.daily_snapshot.append({
            "date": date, "cash": state.cash,
            "position_value": position_value, "netliq": netliq,
            "peak_nav": state.peak_nav, "dd": dd_eod,
            "n_positions": len(state.positions),
            "weights": weights,
        })

    # Force-exit any remaining positions on final bar
    final_date = all_dates[-1]
    for ticker in list(state.positions.keys()):
        pos = state.positions.pop(ticker)
        c = cache[ticker]
        ti = ticker_idx[ticker].get(final_date)
        if ti is None:
            continue
        exit_px = c["close"][ti] * (1.0 - slip)
        exit_pct = (exit_px / pos.entry_price - 1.0) * 100.0
        mae_final = min(pos.mae_pct, exit_pct)
        mfe_final = max(pos.mfe_pct, exit_pct)
        is_ratio_t = bool(c.get("is_ratio", False))
        leg_price_arr = c.get("leg_price")
        exit_leg_price = (
            float(leg_price_arr[ti]) if leg_price_arr is not None else 0.0
        )
        entry_notional = pos.shares * pos.entry_price
        exit_notional = pos.shares * exit_px
        entry_comm = commission_per_side(
            pos.shares, entry_notional,
            model=commission_model, flat_tx_cost=tx_cost,
            is_ratio=is_ratio_t, leg_price=pos.entry_leg_price,
        )
        exit_comm = commission_per_side(
            pos.shares, exit_notional,
            model=commission_model, flat_tx_cost=tx_cost,
            is_ratio=is_ratio_t, leg_price=exit_leg_price,
        )
        commission = entry_comm + exit_comm
        proceeds = pos.shares * exit_px - exit_comm
        state.cash += proceeds
        state.trades.append(TradeLog(
            ticker=ticker, entry_date=pos.entry_date,
            entry_price=pos.entry_price, exit_date=final_date,
            exit_price=exit_px, shares=pos.shares,
            pnl_dollars=proceeds - pos.shares * pos.entry_price,
            pnl_pct=(exit_px / pos.entry_price - 1.0) * 100,
            exit_reason="force_eow",
            entry_pct_cash=pos.entry_pct_cash,
            entry_pct_nav=pos.entry_pct_nav,
            mae_pct=mae_final, mfe_pct=mfe_final,
            hold_days=(final_date - pos.entry_date).days,
            commission_dollars=commission,
        ))

    return state
