"""Static base-weight maps + scheme-name parser + ``build_oracle`` factory.

Extracted from
``obv_pivot/portfolio_analysis/scripts/no_rebalance_replay.py`` lines
~131-385 (2026-04-30). Strategy-agnostic.

Canonical schemes follow the 4-layer model from
``docs/PORTFOLIO_WEIGHTING_METHODS.md`` §1.

**Cap-range widened 2026-05-18** (ref-doc Q4 / Q12, c39): canonical
``DYNAMIC_CAPS = (0.05, 0.10, 0.20, 0.30, 0.50)`` — 5 caps. Drops 0.15
(rare bind-point), adds 0.30 + 0.50 for very-sparse strategies. With
n_active=1 + cap=0.50, a single signal can take up to ~47.5% of cash
on entry (cap × buffer 0.95). Effective concentration is bounded by
the no-touch incumbents invariant: subsequent signals consume cash
sequentially under ``cash_fraction_seq_capped``.

**Top-N parser removed 2026-05-18** (ref-doc Q9, c36) — ``_top<N>``
suffixes no longer parse; raise ValueError. Active runs that previously
used top-N (e.g., ``sharpe_weighted_top10``) are not re-loadable here.
Use the underlying cap variant instead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from strategy_tester.s5_replay.walker import ANNUALIZE

# ---------------------------------------------------------------------------
# Cap series + scheme registry
# ---------------------------------------------------------------------------

# Wide-sweep cap range (2026-05-18). 0.05/0.10 cover dense cohorts;
# 0.20/0.30/0.50 cover sparse / post-DSR-0.95 reduced universes.
DYNAMIC_CAPS = (0.05, 0.10, 0.20, 0.30, 0.50)
ROLL_FAMILIES: tuple[str, ...] = (
    "roll_sr", "roll_iv", "roll_sortino", "roll_calmar", "roll_mom", "roll_er",
)


def all_scheme_names() -> list[str]:
    """Canonical scheme list — total = **59** (2026-05-18).

    Breakdown:
    - 4  static: equal_weight, sharpe_weighted, inverse_vol, hrp
    - 5  flat_1k_capN — cap{05,10,20,30,50}
    - 15 weighted_dyn: {equal,sharpe_wt,inv_vol}_dyn_cap{05,10,20,30,50}
    - 5  hrp_capN
    - 30 rolling-online: 6 families × 5 caps —
      roll_sr / roll_iv / roll_sortino / roll_calmar / roll_mom / roll_er
      each with cap{05,10,20,30,50}

    Top-N variants permanently dropped (2026-05-18 per ref-doc Q9, c36).

    Online variants (`roll_*`) rank by per-bar rolling window with
    `.shift(1)` upstream to prevent look-ahead.
    """
    names = ["equal_weight", "sharpe_weighted", "inverse_vol", "hrp"]
    for cap in DYNAMIC_CAPS:
        names.append(f"flat_1k_cap{int(cap * 100):02d}")
    for wmap in ("equal", "sharpe_wt", "inv_vol"):
        for cap in DYNAMIC_CAPS:
            names.append(f"{wmap}_dyn_cap{int(cap * 100):02d}")
    for cap in DYNAMIC_CAPS:
        names.append(f"hrp_cap{int(cap * 100):02d}")
    for fam in ROLL_FAMILIES:
        for cap in DYNAMIC_CAPS:
            names.append(f"{fam}_cap{int(cap * 100):02d}")
    return names


def parse_scheme(name: str) -> dict:
    """Decode a scheme name → {family, base_map, cap}.

    family ∈ {"static", "flat_1k", "weighted_dyn", + roll_* families}.
    base_map ∈ {"equal","sharpe","inv_vol","hrp","sharpe_wt", roll_* names}.
    cap = float in [0,1] or None.

    Top-N suffixes (``_top<N>``) raise ``ValueError`` as of 2026-05-18
    (ref-doc Q9, c36 — parser support removed entirely).
    """
    # Strip sizing-rule prefix (cf__/cfc__/seq__/ps__) — the runner tags
    # cross-sizing-rule schemes this way; the underlying scheme spec is
    # prefix-invariant.
    for sr_prefix in ("cfc__", "cf__", "seq__", "plg__"):
        if name.startswith(sr_prefix):
            name = name[len(sr_prefix):]
            break
    # Reject _top<N> suffixes (parser support removed 2026-05-18, ref-doc Q9).
    if "_top" in name:
        tail = name.rsplit("_top", 1)[-1]
        if tail.isdigit():
            raise ValueError(
                f"top_n parser support removed 2026-05-18 (ref-doc Q9, c36). "
                f"Scheme {name!r} no longer parses. Use the underlying "
                f"cap variant (e.g., 'sharpe_weighted' or '<base>_cap10')."
            )

    if name in {"equal_weight", "sharpe_weighted", "inverse_vol", "hrp"}:
        base = {
            "equal_weight": "equal",
            "sharpe_weighted": "sharpe",
            "inverse_vol": "inv_vol",
            "hrp": "hrp",
        }[name]
        return {"family": "static", "base_map": base, "cap": None}
    # Adapter-defined schemes (no lib oracle equivalent — built locally
    # by the strategy adapter, e.g. fred regime_portfolio uses conviction
    # multiplicatively on top of every canonical scheme: weight ∝
    # canonical_rank × conviction, normalized, with canonical's cap on top.
    # ``conv_<canonical_name>`` strips the prefix and recurses so
    # build_html / scheme_per_name_cap inherit the canonical's cap value.
    if name.startswith("conv_"):
        return parse_scheme(name[len("conv_"):])
    if name.startswith("flat_1k_cap"):
        return {
            "family": "flat_1k",
            "base_map": "equal",
            "cap": int(name[-2:]) / 100.0,
        }
    for fam in ROLL_FAMILIES:
        prefix = f"{fam}_cap"
        if name.startswith(prefix):
            return {
                "family": fam, "base_map": fam,
                "cap": int(name[-2:]) / 100.0,
            }
    for prefix in ("equal_dyn_cap", "sharpe_wt_dyn_cap", "inv_vol_dyn_cap"):
        if name.startswith(prefix):
            base = prefix.replace("_dyn_cap", "")
            return {
                "family": "weighted_dyn",
                "base_map": base,
                "cap": int(name[-2:]) / 100.0,
            }
    # HRP cap variants — added 2026-05-17 (cap-sweep completeness).
    # Routes to weighted_dyn family using the precomputed "hrp" base map.
    if name.startswith("hrp_cap"):
        return {
            "family": "weighted_dyn",
            "base_map": "hrp",
            "cap": int(name[-2:]) / 100.0,
        }
    raise ValueError(f"Unknown scheme: {name}")


def scheme_per_name_cap(scheme: str, default: float) -> float:
    """Resolve per-name cap from a scheme name. ``*_cap05`` → 0.05 etc.
    Static / no-suffix schemes use ``default``."""
    spec = parse_scheme(scheme)
    cap = spec.get("cap")
    return float(cap) if cap is not None else float(default)


# ---------------------------------------------------------------------------
# Static base-weight maps (computed once, mirror build_portfolio.py:682-700)
# ---------------------------------------------------------------------------


def equal_weight_map(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def sharpe_weighted_map(returns_df: pd.DataFrame) -> dict[str, float]:
    cols = list(returns_df.columns)
    arr = returns_df.to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    std_safe = np.where(std > 1e-12, std, np.nan)
    sharpes = np.nan_to_num((mean / std_safe) * np.sqrt(ANNUALIZE), nan=0.0)
    pos = np.clip(sharpes, 0.0, None)
    total = float(pos.sum())
    if total < 1e-12:
        return equal_weight_map(cols)
    w = pos / total
    return {t: float(w[i]) for i, t in enumerate(cols)}


def inv_vol_map(returns_df: pd.DataFrame) -> dict[str, float]:
    cols = list(returns_df.columns)
    arr = returns_df.to_numpy(dtype=np.float64)
    vols = np.nanstd(arr, axis=0, ddof=1)
    valid = vols[vols > 1e-10]
    median_vol = float(np.median(valid)) if valid.size > 0 else 0.01
    if not np.isfinite(median_vol) or median_vol < 1e-10:
        median_vol = 0.01
    vols_filled = np.where(np.isfinite(vols) & (vols > 1e-10), vols, median_vol)
    inv = 1.0 / np.clip(vols_filled, 1e-10, None)
    w = inv / inv.sum()
    return {t: float(w[i]) for i, t in enumerate(cols)}


def hrp_map(returns_df: pd.DataFrame) -> dict[str, float]:
    """LdP AFML Ch.16 — copied from build_portfolio.py:209-259."""
    cols = list(returns_df.columns)
    n = len(cols)
    if n == 1:
        return {cols[0]: 1.0}
    if len(returns_df) < 100:
        return equal_weight_map(cols)
    corr = returns_df.corr().fillna(0.0).to_numpy()
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    try:
        linkage = sch.linkage(ssd.squareform(dist, checks=False), method="single")
        order = sch.leaves_list(linkage)
    except Exception:
        return equal_weight_map(cols)
    cov = returns_df.cov().to_numpy()
    weights = np.ones(n)
    clusters = [list(order)]
    while clusters:
        new_clusters = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]

            def _cv(subset: list[int]) -> float:
                c = cov[np.ix_(subset, subset)]
                diag = np.diag(c)
                iv = 1.0 / np.clip(diag, 1e-10, None)
                w = iv / iv.sum()
                return float(w @ c @ w)

            cv_l, cv_r = _cv(left), _cv(right)
            alpha = 0.5 if cv_l + cv_r < 1e-20 else 1 - cv_l / (cv_l + cv_r)
            for j in left:
                weights[j] *= alpha
            for j in right:
                weights[j] *= 1 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    if weights.sum() < 1e-12:
        return equal_weight_map(cols)
    weights = weights / weights.sum()
    return {t: float(weights[i]) for i, t in enumerate(cols)}


# ---------------------------------------------------------------------------
# Weight oracles
# ---------------------------------------------------------------------------


def renorm_cap_renorm(
    base: dict[str, float], cap: float | None,
) -> dict[str, float]:
    """LEGACY — renorm → cap → renorm on an active set's base weights.

    Status (2026-05-17): retained for back-compat with all 9 deployable v2
    strategies (2026-06-01 kickoff cohort). Do NOT use for new strategies —
    the second renorm cancels the cap when base weights are symmetric
    (proof: any uniform input × any cap → uniform output). See
    `PORTFOLIO_WEIGHTING_METHODS.md §4.8` and the 2026-05-17 review.

    For new strategies / new sweeps use `absolute_cap_truncate` (below).
    """
    if not base:
        return {}
    total = sum(base.values())
    if total <= 0:
        return {}
    w = {t: v / total for t, v in base.items()}
    if cap is None:
        return w
    w = {t: min(v, cap) for t, v in w.items()}
    total = sum(w.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in w.items()}


def absolute_cap_truncate(
    base: dict[str, float], cap: float | None,
) -> dict[str, float]:
    """Paleologo APM §3.6 absolute cap — single renorm + truncate (no 2nd renorm).

    Status (2026-05-17): RECOMMENDED for new strategies. The residual
    ``1 - Σ w_capped`` is deliberately uncalled — it stays in cash when the
    walker sizes positions, giving the cap real dollar-space effect.

    Behavioural contrast with `renorm_cap_renorm` (legacy):

    * n_active=1, cap=0.20:
        - legacy:    {sole: 1.0}    (cap cancelled by 2nd renorm)
        - this fn:   {sole: 0.20}   (cap binds; 0.80 stays in cash)
    * n_active=5 uniform, cap=0.10:
        - legacy:    {each: 0.20}   (cap cancelled — same as no cap)
        - this fn:   {each: 0.10}   (Σ=0.50; 0.50 stays in cash)
    * n_active=5 skewed [.5,.3,.1,.07,.03], cap=0.20:
        - legacy:    [.357,.286,.143,.100,.043]   (caps then renorms)
        - this fn:   [.200,.200,.100,.070,.030]   (Σ=0.60; 0.40 to cash)

    Source: Paleologo APM §3.6 single-stock cap principle (formula
    ``cap %GMV = (CPR² − 1) / N_universe``). The decision to leave the
    residual as cash (rather than redistribute, as in legacy) is in-house
    — no published precedent — but matches Sinclair OTPV Ch.9 gambler's-ruin
    principle (survival before geometric growth) and Carver qoppac 2016-03
    minimum-N-contracts-at-max-forecast spirit.
    """
    if not base:
        return {}
    total = sum(base.values())
    if total <= 0:
        return {}
    w = {t: v / total for t, v in base.items()}
    if cap is None:
        return w
    return {t: min(v, cap) for t, v in w.items()}


def _make_date_aware_oracle(
    scores_df: "pd.DataFrame",
    cap: float | None,
) -> Callable[[set[str], pd.Timestamp], dict[str, float]]:
    """Generic date-aware oracle for any rolling-score family.

    At bar t, looks up `scores_df.loc[t]` (or last available row before t) to
    rank/weight active set. Caller is responsible for `.shift(1)` on scores_df
    to prevent look-ahead. Negative scores → 0 weight (uniform fallback when
    all scores ≤ 0). Cap applied via renorm-cap-renorm.

    Used by all `roll_*` families: roll_sr, roll_iv, roll_sortino, roll_calmar,
    roll_mom, roll_er.
    """
    def oracle(active: set[str], date: pd.Timestamp) -> dict[str, float]:
        if not active:
            return {}
        if date in scores_df.index:
            row = scores_df.loc[date]
        else:
            idx = int(np.searchsorted(
                scores_df.index.values, np.datetime64(date),
            ))
            if idx == 0:
                return renorm_cap_renorm({t: 1.0 for t in active}, cap)
            row = scores_df.iloc[idx - 1]
        sub: dict[str, float] = {}
        for t in active:
            v = float(row.get(t, 0.0)) if t in row.index else 0.0
            sub[t] = max(0.0, v)
        if sum(sub.values()) < 1e-12:
            return renorm_cap_renorm({t: 1.0 for t in active}, cap)
        return renorm_cap_renorm(sub, cap)
    return oracle


def build_oracle(
    scheme: str,
    base_maps: dict[str, dict[str, float]],
    rolling_sr: pd.DataFrame | None,
    rolling_iv: pd.DataFrame | None = None,
    rolling_sortino: pd.DataFrame | None = None,
    rolling_calmar: pd.DataFrame | None = None,
    rolling_mom: pd.DataFrame | None = None,
    rolling_er: pd.DataFrame | None = None,
) -> Callable[[set[str], pd.Timestamp], dict[str, float]]:
    """Return a callable(active_set, date) -> {ticker: weight}.

    All schemes return the FULL active-set weight vector; callers pull the
    entries they care about. For "static" schemes we still use cap=None and
    renormalise across the active set so behaviour matches the dynamic case
    of "this is what we sized at entry time given who was active right then."
    """
    spec = parse_scheme(scheme)
    family = spec["family"]
    base = spec["base_map"]
    cap = spec["cap"]

    if family == "static":
        # Static means "fixed per-ticker target weight"; under no-rebalance we
        # apply the constant base map normalised over the active set, no cap.
        bm = base_maps[base]

        def oracle(active: set[str], _date: pd.Timestamp) -> dict[str, float]:
            sub = {t: bm.get(t, 0.0) for t in active}
            return renorm_cap_renorm(sub, None)
        return oracle

    if family == "flat_1k":
        # Equal across active, cap+renorm.
        def oracle(active: set[str], _date: pd.Timestamp) -> dict[str, float]:
            if not active:
                return {}
            sub = {t: 1.0 for t in active}
            return renorm_cap_renorm(sub, cap)
        return oracle

    if family == "weighted_dyn":
        bm = base_maps[base]

        def oracle(active: set[str], _date: pd.Timestamp) -> dict[str, float]:
            sub = {t: bm.get(t, 0.0) for t in active}
            return renorm_cap_renorm(sub, cap)
        return oracle

    # All rolling-score families share the same date-aware oracle skeleton.
    # The only difference is the score matrix (rolling_sr / rolling_iv / etc.)
    # passed to `_make_date_aware_oracle`. Caller is responsible for `.shift(1)`
    # to prevent look-ahead.
    _roll_score_map: dict[str, "pd.DataFrame | None"] = {
        "roll_sr": rolling_sr,
        "roll_iv": rolling_iv,
        "roll_sortino": rolling_sortino,
        "roll_calmar": rolling_calmar,
        "roll_mom": rolling_mom,
        "roll_er": rolling_er,
    }
    if family in _roll_score_map:
        scores = _roll_score_map[family]
        if scores is None:
            raise ValueError(
                f"{family!r} family requires `{family.replace('roll_', 'rolling_')}` "
                f"argument to build_oracle. See "
                f"`cum_rsi_v2/portfolio_analysis/scripts/no_rebalance_replay.py::"
                f"build_rolling_*` for the canonical computation patterns."
            )
        return _make_date_aware_oracle(scores, cap)

    raise ValueError(f"Unsupported family: {family}")
