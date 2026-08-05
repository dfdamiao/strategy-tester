"""Grading must never be cheaper than screening.

Ported from the monorepo audit finding, 2026-07-26. Two engines, two cost
models:

  * S2/S3 SELECTION — the validators charge a flat ``cost_per_side=0.001``
    (10 bps) per side (see e.g. ``s3_validation/wfa_rolling.py``). Flat
    because at that stage there is no NAV and no share count, so a
    per-share schedule is literally not computable.
  * S5 GRADING — ``s5_replay/walker.py``, IBKR Pro Fixed ($0.005/share,
    min $1.00, capped at 1% of notional) plus ``SLIP_BPS``.

At a realistic entry size (~$28.5k on $100k) the commission is ~0.5 bps, so
with the old 5 bps slippage grading cost 5.5 bps/side against a 10 bps
screen. Units were screened under harsher costs than they were graded under
and every headline was flattered by ~9 bps per round trip.

``SLIP_BPS`` was raised to 9.5 rather than dropping the screen to match,
because lowering the screen loosens selection and moves cohort membership;
raising slippage leaves membership untouched and only makes the reported
number more conservative. This test pins that ordering so it cannot
silently invert again.

Note the relationship is ticket-size dependent: below ~$2,000 per position
the $1 commission minimum makes grading DEARER than the screen regardless
of slippage. The invariant asserted here is one-sided (grading >= screen),
which holds at every size.
"""

from __future__ import annotations

import pytest

from strategy_tester.s5_replay.walker import SLIP_BPS, commission_per_side

pytestmark = pytest.mark.unit

# The flat screen cost the S2/S3 engines charge: cost_per_side=0.001 default
# (10 bps/side, Chan AT 2013 Ch.3) across the validators.
SCREEN_COSTS_BPS = 10.0

# Realistic entry sizes. The lower end is the measured p95 for the sweeps;
# the upper end is a concentrated cap20/cap30 book on a $100k NAV.
REALISTIC_POSITIONS = (2_000.0, 5_000.0, 10_000.0, 20_000.0, 28_500.0, 50_000.0)
PRICE = 100.0


def _grading_bps_per_side(notional: float, price: float = PRICE) -> float:
    shares = max(1, int(notional / price))
    comm = commission_per_side(shares, notional, model="ibkr_pro_fixed")
    return (comm / notional + SLIP_BPS) * 1e4


@pytest.mark.parametrize("notional", REALISTIC_POSITIONS)
def test_grading_is_never_cheaper_than_screening(notional: float) -> None:
    """The invariant. A unit must not be graded under kinder costs than the
    ones it had to survive to be selected."""
    grading = _grading_bps_per_side(notional)
    assert grading >= SCREEN_COSTS_BPS - 1e-9, (
        f"at a ${notional:,.0f} position grading costs {grading:.2f} bps/side "
        f"but selection charges {SCREEN_COSTS_BPS:.2f} — the headline is "
        f"flattered by {2 * (SCREEN_COSTS_BPS - grading):.1f} bps per round "
        f"trip. Raise SLIP_BPS; do NOT lower the screen cost, which would "
        f"loosen selection and move cohort membership."
    )


def test_slippage_is_the_conservative_value_not_the_old_optimistic_one() -> None:
    """Guards the specific regression: SLIP_BPS back at 5 bps re-opens the gap."""
    assert SLIP_BPS * 1e4 >= 9.5 - 1e-9, (
        f"SLIP_BPS is {SLIP_BPS * 1e4:.1f} bps. At the measured ~$28.5k entry "
        f"size the IBKR commission is only ~0.5 bps, so anything below ~9.5 "
        f"puts grading under the 10 bps screen."
    )


def test_the_gap_is_actually_closed_at_the_measured_entry_size() -> None:
    """p95_entry_pct_cash ~ 28.5% of a $100k book — the size that mattered."""
    assert _grading_bps_per_side(28_500.0) == pytest.approx(10.0, abs=0.15)


def test_small_tickets_were_already_dearer_not_cheaper() -> None:
    """Documents the ticket-size dependence: below ~$2k the $1 commission
    minimum dominates, so grading was ALREADY stricter there. The bug was
    never universal, which is why it survived — it only bit at realistic
    sizes."""
    tiny = _grading_bps_per_side(500.0)
    assert tiny > SCREEN_COSTS_BPS * 2
