"""Tests for the 2026-05-17 cap-mechanism redesign.

Covers:
  1. `oracles.absolute_cap_truncate` — new Pattern A cap (single renorm,
     truncate, residual stays uncalled — vs. legacy `renorm_cap_renorm`
     which 2-renorms and cancels the cap on uniform input).
  2. `walker.cash_fraction_capped` — new sizing rule that mirrors
     `paleologo_strict` minus the DD throttle. Cap binds in dollar space.
  3. `oracles.hrp_cap{N}` parser — newly added cap-sweep variants for HRP.

Reference review:
  docs/PORTFOLIO_CONSTRUCTION.md §1–§2 (locked goals + 21 decisions;
    merged 2026-05-18 from former PORTFOLIO_REDESIGN_SCOPING.html 2026-05-17)
  PORTFOLIO_WEIGHTING_METHODS.md §4 (Layer 3 cash) + §4.7 Paleologo cap
  Memory: feedback_dsr_correct_inputs_per_asset.md (cap mechanism context)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s5_replay.oracles import (
    absolute_cap_truncate,
    all_scheme_names,
    parse_scheme,
    renorm_cap_renorm,
)
from strategy_tester.s5_replay.walker import (
    CASH_BUFFER,
    DEFAULT_SEED_NAV,
    SIZING_RULES,
    walk_portfolio_oracle,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. absolute_cap_truncate — Pattern A semantics
# ---------------------------------------------------------------------------


def test_absolute_cap_truncate_n_active_1_cap_binds() -> None:
    """At n_active=1 with cap=0.20, the lone weight must be truncated to 0.20.
    Residual (0.80) is implied — caller leaves it as cash. Contrast with
    legacy renorm_cap_renorm which restores 1.0 via 2nd renorm.
    """
    out = absolute_cap_truncate({"SPY": 1.0}, cap=0.20)
    assert out == {"SPY": 0.20}
    # Legacy comparison: cap is cancelled
    legacy = renorm_cap_renorm({"SPY": 1.0}, cap=0.20)
    assert legacy == {"SPY": 1.0}


def test_absolute_cap_truncate_uniform_base_cap_binds_at_each_name() -> None:
    """Uniform base [1/5 each], cap=0.10. Each truncates to 0.10.
    Σ output = 0.50 (NOT renormalised back to 1.0).
    """
    base = {f"T{i}": 1.0 for i in range(5)}
    out = absolute_cap_truncate(base, cap=0.10)
    assert all(abs(v - 0.10) < 1e-9 for v in out.values())
    assert abs(sum(out.values()) - 0.50) < 1e-9
    # Legacy: 2nd renorm restores 1/5 each (cap no-op)
    legacy = renorm_cap_renorm(base, cap=0.10)
    assert all(abs(v - 0.20) < 1e-9 for v in legacy.values())
    assert abs(sum(legacy.values()) - 1.0) < 1e-9


def test_absolute_cap_truncate_skewed_base_asymmetric_truncation() -> None:
    """Skewed base — heavy names hit cap, light names stay.
    Source weights [0.5, 0.3, 0.1, 0.07, 0.03] (Σ=1.0), cap=0.20.
    After: [0.20, 0.20, 0.10, 0.07, 0.03], Σ=0.60, residual 0.40 to cash.
    """
    base = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.07, "E": 0.03}
    out = absolute_cap_truncate(base, cap=0.20)
    assert out["A"] == pytest.approx(0.20)
    assert out["B"] == pytest.approx(0.20)
    assert out["C"] == pytest.approx(0.10)
    assert out["D"] == pytest.approx(0.07)
    assert out["E"] == pytest.approx(0.03)
    assert sum(out.values()) == pytest.approx(0.60)


def test_absolute_cap_truncate_cap_none_passthrough() -> None:
    """cap=None means: just renormalize once, no truncation."""
    out = absolute_cap_truncate({"A": 2.0, "B": 8.0}, cap=None)
    assert out == {"A": 0.20, "B": 0.80}


def test_absolute_cap_truncate_empty_input() -> None:
    """Empty input → empty output."""
    assert absolute_cap_truncate({}, cap=0.10) == {}
    assert absolute_cap_truncate({"A": 0.0, "B": 0.0}, cap=0.10) == {}


def test_absolute_cap_truncate_residual_invariant() -> None:
    """Σ output ≤ 1.0 always. Residual = 1 − Σ output stays in cash."""
    for n in (1, 2, 5, 10):
        for cap in (0.05, 0.10, 0.20, 0.50):
            base = {f"T{i}": 1.0 for i in range(n)}
            out = absolute_cap_truncate(base, cap)
            assert sum(out.values()) <= 1.0 + 1e-9
            assert all(v <= cap + 1e-9 for v in out.values())


# ---------------------------------------------------------------------------
# 2. hrp_cap{N} parser — added 2026-05-17
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap_pct,expected_cap", [
    ("05", 0.05), ("10", 0.10), ("20", 0.20), ("30", 0.30), ("50", 0.50),
])
def test_hrp_cap_parses_to_weighted_dyn(cap_pct: str, expected_cap: float) -> None:
    spec = parse_scheme(f"hrp_cap{cap_pct}")
    assert spec["family"] == "weighted_dyn"
    assert spec["base_map"] == "hrp"
    assert spec["cap"] == pytest.approx(expected_cap)
    assert "top_n" not in spec  # top_n parser removed 2026-05-18 (ref-doc Q9)


def test_hrp_cap_in_canonical_list() -> None:
    """All 5 hrp_cap{N} variants must appear in the canonical scheme list.

    Cap range widened to {05,10,20,30,50} 2026-05-18 (ref-doc Q4/Q12).
    """
    names = all_scheme_names()
    for cap_pct in ("05", "10", "20", "30", "50"):
        assert f"hrp_cap{cap_pct}" in names, (
            f"hrp_cap{cap_pct} missing from all_scheme_names()"
        )


# ---------------------------------------------------------------------------
# 3. cash_fraction_capped sizing rule registered + accepted by walker
# ---------------------------------------------------------------------------


def test_cash_fraction_capped_in_sizing_rules() -> None:
    assert "cash_fraction_capped" in SIZING_RULES


def _make_minimal_cache(close: float = 100.0, n_bars: int = 30) -> dict:
    """Tiny single-asset cache with one entry signal at bar 5."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    pos = np.zeros(n_bars, dtype=np.int8)
    pos[5:] = 1  # enter at bar 5, hold to end
    return {
        "SPY": {
            "index": idx,
            "close": np.full(n_bars, close, dtype=np.float64),
            "low": np.full(n_bars, close * 0.99, dtype=np.float64),
            "atr": np.ones(n_bars, dtype=np.float64),
            "pos_raw": pos,
            "stop_code": 0,
            "stop_param": 0.0,
            "base_weight": 1.0,
        }
    }


def test_cash_fraction_capped_caps_at_per_name_cap() -> None:
    """Under cash_fraction_capped at n_active=1 with per_name_cap=0.20,
    entry must size at ≤ 20% of cash × buffer (NOT 95% as under
    cash_fraction).
    """
    cache = _make_minimal_cache(close=100.0, n_bars=30)
    seed = DEFAULT_SEED_NAV  # $100k
    state = walk_portfolio_oracle(
        cache,
        weight_oracle=lambda active, d: {t: 1.0 for t in active},
        seed_nav=seed,
        sizing_rule="cash_fraction_capped",
        per_name_cap=0.20,
    )
    # Exactly one entry on bar 5
    assert len(state.trades) + len(state.positions) >= 1
    # Capture entry diagnostics — under cash_fraction_capped at cap=0.20,
    # cost / cash_at_bar_start should be at most cap × buffer × 100 ≈ 19%
    cap_pct = 0.20 * CASH_BUFFER * 100.0  # = 19.0
    if state.positions:
        pos = next(iter(state.positions.values()))
        assert pos.entry_pct_cash <= cap_pct + 0.1, (
            f"entry_pct_cash={pos.entry_pct_cash:.2f}% exceeds cap×buffer="
            f"{cap_pct:.2f}%"
        )
    if state.trades:
        tr = state.trades[0]
        assert tr.entry_pct_cash <= cap_pct + 0.1


def test_cash_fraction_unchanged_n_active_1_still_95pct() -> None:
    """Regression: cash_fraction (legacy live default) must STILL deploy
    ~95% at n_active=1. This is the bug the new rule fixes, but legacy
    behavior must not shift for the 9 locked strategies.
    """
    cache = _make_minimal_cache(close=100.0, n_bars=30)
    state = walk_portfolio_oracle(
        cache,
        weight_oracle=lambda active, d: {t: 1.0 for t in active},
        seed_nav=DEFAULT_SEED_NAV,
        sizing_rule="cash_fraction",
    )
    # Should deploy ~95% (cap not enforced)
    if state.positions:
        pos = next(iter(state.positions.values()))
        assert pos.entry_pct_cash > 90.0, (
            f"cash_fraction at n_active=1 must deploy >90%; got "
            f"{pos.entry_pct_cash:.2f}%"
        )
    elif state.trades:
        tr = state.trades[0]
        assert tr.entry_pct_cash > 90.0
    else:
        pytest.fail("No entry observed — fixture broken")


def test_cash_fraction_capped_residual_stays_in_cash() -> None:
    """When cap binds, the residual (1 - cap×buffer) stays as cash —
    NOT redistributed to other names. Critical for the "reserve cash for
    future entries" goal."""
    cache = _make_minimal_cache(close=100.0, n_bars=30)
    seed = DEFAULT_SEED_NAV
    state = walk_portfolio_oracle(
        cache,
        weight_oracle=lambda active, d: {t: 1.0 for t in active},
        seed_nav=seed,
        sizing_rule="cash_fraction_capped",
        per_name_cap=0.20,
    )
    # After entry at bar 5, on bar 6 (next snapshot) cash should be
    # roughly seed × (1 - 0.20 × buffer) ≈ $80,950
    snap = pd.DataFrame(state.daily_snapshot)
    post_entry = snap[snap["date"] >= cache["SPY"]["index"][5]]
    assert len(post_entry) >= 1
    expected_remaining = seed * (1.0 - 0.20 * CASH_BUFFER)
    actual_cash = float(post_entry.iloc[0]["cash"])
    # Allow ±5% tolerance (slippage + tx_cost rounding)
    assert actual_cash >= expected_remaining * 0.95, (
        f"cash {actual_cash:.0f} should be >= ~{expected_remaining:.0f}"
    )
