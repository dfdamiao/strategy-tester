"""OBV-pivot signal (Granville 1963 + Dow structure).

Computes On-Balance Volume on close+volume, detects pivots on the OBV series,
labels each pivot HH / LH / HL / LL vs the previous `n_compare` same-type
pivot, then maps the pivot stream to a 0/1 long position via one of three
state machines.

Three pivot-detection methods (ported from
`technical_analysis/obv_signal_study/obv_pivot_study/pivots.py`):

- ``fractal``       (Bulkowski 2005) — centered rolling max/min,
                      confirms at ``pivot_idx + swing_length/2``.
- ``topological``   (Edelsbrunner & Harer 2008) — top-K peaks by 1D
                      topological persistence (scipy ``find_peaks``
                      prominence), re-run on expanding window every 5 bars;
                      each pivot is recorded at its first-appearance day.
- ``pip``           (Fu et al. 2007) — Perceptually Important Points;
                      iterative max-perpendicular-distance segmentation,
                      expanding-window re-run like topological.

Three mappings (ported from
`technical_analysis/obv_signal_study/obv_pivot_study/signals.py`):

- ``A``  HL -> long, LH -> flat (Dow / SMC classical)
- ``B``  HH -> long, LL -> flat (breakout / momentum)
- ``C``  first HL after any LL -> long,
          first LH after any HH -> flat  (SMC Change-of-Character)

Execution convention (same-bar close + 5 bps slippage) and stop overlays
(fixed % and ATR-trail) are not handled here — ``precompute`` returns the
raw pivot timeline and a pre-stop 0/1 position series; the S2 walker
in ``lib/s2_optimize/grid_obv_pivot.py`` applies stops and fills.

References:
    Granville 1963 — OBV definition
    Rhea 1932 (Dow Theory) — HH/LH/HL/LL structure
    Bulkowski 2005 — Fractal pivots
    Edelsbrunner & Harer 2008 — Topological persistence
    Fu et al. 2007 — Perceptually Important Points
    Lin 2013 (SMC) — CHoCH
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)

PIVOT_METHODS = ("fractal", "topological", "pip")
MAPPINGS = ("A", "B", "C")
EXPANDING_STEP = 5
EXPANDING_WARMUP = 120


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Granville 1963 OBV: cumulative signed volume.

    ``obv[0] = 0``; for ``t >= 1``, ``obv[t] = obv[t-1] + sign(dclose[t]) * vol[t]``.
    Returned as a ``pd.Series`` on the close index.
    """
    c = close.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)
    d = np.zeros_like(c)
    d[1:] = np.sign(c[1:] - c[:-1]) * v[1:]
    return pd.Series(np.cumsum(d), index=close.index)


# ---------------------------------------------------------------------------
# Pivot detectors (operate on any 1D series)
# ---------------------------------------------------------------------------


def _fractal(series: pd.Series, swing_length: int) -> tuple[list[int], list[int]]:
    window = max(3, int(swing_length))
    rmax = series.rolling(window, center=True, min_periods=window).max()
    rmin = series.rolling(window, center=True, min_periods=window).min()
    arr = series.to_numpy()
    tops = np.where(arr == rmax.to_numpy())[0].tolist()
    bottoms = np.where(arr == rmin.to_numpy())[0].tolist()
    return tops, bottoms


def _topological(series: pd.Series, top_k: int) -> tuple[list[int], list[int]]:
    arr = series.to_numpy(dtype=float)
    if len(arr) < 3:
        return [], []

    def _topk(a: np.ndarray, k: int) -> list[int]:
        peaks, props = find_peaks(a, prominence=1e-12)
        if len(peaks) == 0:
            return []
        proms = np.asarray(props["prominences"])
        order = np.argsort(-proms)[:k]
        return sorted(int(peaks[i]) for i in order)

    return _topk(arr, int(top_k)), _topk(-arr, int(top_k))


def _pip(series: pd.Series, k: int) -> tuple[list[int], list[int]]:
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    k = int(k)
    if n < 2 or k < 3:
        return [], []
    pips: list[int] = [0, n - 1]
    while len(pips) < k:
        best_dist = 0.0
        best_idx = -1
        best_insert = 0
        for j in range(len(pips) - 1):
            a_i, b_i = pips[j], pips[j + 1]
            if b_i - a_i < 2:
                continue
            slope = (arr[b_i] - arr[a_i]) / max(b_i - a_i, 1)
            xs = np.arange(a_i + 1, b_i)
            line_y = arr[a_i] + slope * (xs - a_i)
            dist = np.abs(arr[a_i + 1:b_i] - line_y)
            if len(dist) == 0:
                continue
            loc = int(np.argmax(dist))
            if dist[loc] > best_dist:
                best_dist = float(dist[loc])
                best_idx = a_i + 1 + loc
                best_insert = j + 1
        if best_idx < 0:
            break
        pips.insert(best_insert, best_idx)
    tops: list[int] = []
    bottoms: list[int] = []
    for i in pips[1:-1]:
        if i <= 0 or i >= n - 1:
            continue
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            tops.append(i)
        elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            bottoms.append(i)
    return sorted(tops), sorted(bottoms)


_DETECTORS = {
    "fractal": _fractal,
    "topological": _topological,
    "pip": _pip,
}


# ---------------------------------------------------------------------------
# Classification (HH / LH / HL / LL)
# ---------------------------------------------------------------------------


def classify_pivots(
    tops: list[int],
    bottoms: list[int],
    series: pd.Series,
    n_compare: int = 1,
) -> dict[str, list[int]]:
    arr = series.to_numpy()
    hh: list[int] = []
    lh: list[int] = []
    hl: list[int] = []
    ll: list[int] = []

    t_sorted = sorted(tops)
    for i, t_idx in enumerate(t_sorted):
        if i < n_compare:
            continue
        prev_max = float(max(arr[j] for j in t_sorted[i - n_compare:i]))
        if arr[t_idx] > prev_max:
            hh.append(int(t_idx))
        else:
            lh.append(int(t_idx))

    b_sorted = sorted(bottoms)
    for i, b_idx in enumerate(b_sorted):
        if i < n_compare:
            continue
        prev_min = float(min(arr[j] for j in b_sorted[i - n_compare:i]))
        if arr[b_idx] > prev_min:
            hl.append(int(b_idx))
        else:
            ll.append(int(b_idx))

    return {"HH": hh, "LH": lh, "HL": hl, "LL": ll}


# ---------------------------------------------------------------------------
# Causal timeline builders
# ---------------------------------------------------------------------------


def _fractal_timeline(
    obv: pd.Series, swing_length: int, n_compare: int,
) -> list[tuple[int, str]]:
    """Fractal is causal: pivot at bar i is known at i + swing_length/2.

    act_day = i + half + 1 (+1 because fill happens same-bar-close of the
    confirmation bar; half steps forward from the pivot bar to when the
    centered window has closed).
    """
    tops, bottoms = _fractal(obv, swing_length)
    classes = classify_pivots(tops, bottoms, obv, n_compare=n_compare)
    half = swing_length // 2
    out: list[tuple[int, str]] = []
    n = len(obv)
    for ptype in ("HH", "LH", "HL", "LL"):
        for idx in classes[ptype]:
            act_idx = idx + half
            if act_idx < n:
                out.append((int(act_idx), ptype))
    return sorted(out)


def _expanding_timeline(
    obv: pd.Series,
    method: str,
    method_param: float | int,
    n_compare: int,
    warmup: int = EXPANDING_WARMUP,
    step: int = EXPANDING_STEP,
) -> list[tuple[int, str]]:
    """Re-run a non-causal detector on expanding windows; record each pivot
    the first day it appears. Acts at first-detection day (no T+1 shift —
    entry is same-bar close at first detection).
    """
    n = len(obv)
    detector = _DETECTORS[method]
    first_seen: dict[int, int] = {}
    final_type: dict[int, str] = {}
    for t in range(warmup, n, step):
        sub = obv.iloc[: t + 1]
        try:
            tops, bottoms = detector(sub, method_param)
        except Exception:
            continue
        classes = classify_pivots(tops, bottoms, sub, n_compare=n_compare)
        for ptype, idxs in classes.items():
            for pi in idxs:
                if pi not in first_seen:
                    first_seen[pi] = t
                    final_type[pi] = ptype
    out = [
        (int(first_seen[pi]), final_type[pi])
        for pi in first_seen
        if first_seen[pi] < n
    ]
    return sorted(out)


def causal_timeline(
    obv: pd.Series,
    method: str,
    method_param: float | int,
    n_compare: int = 1,
) -> list[tuple[int, str]]:
    if method == "fractal":
        return _fractal_timeline(obv, int(method_param), n_compare)
    if method in ("topological", "pip"):
        return _expanding_timeline(obv, method, method_param, n_compare)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Mapping: pivot timeline -> 0/1 position series
# ---------------------------------------------------------------------------


def _timeline_to_position(
    timeline: list[tuple[int, str]],
    index: pd.DatetimeIndex,
    update_fn,
) -> pd.Series:
    state: dict = {"pos": 0, "recent_LL": False, "recent_HH": False}
    events: dict[int, int] = {0: 0}
    n = len(index)
    for act_idx, ptype in timeline:
        if act_idx >= n:
            continue
        new_pos = update_fn(state, ptype)
        if new_pos is not None:
            state["pos"] = new_pos
            events[act_idx] = new_pos
    pos = pd.Series(index=index, dtype=float)
    for act_idx, p in events.items():
        pos.iloc[act_idx] = p
    return pos.ffill().fillna(0.0)


def _mapping_a(timeline, index):
    def update(_state, ptype):
        if ptype == "HL":
            return 1
        if ptype == "LH":
            return 0
        return None
    return _timeline_to_position(timeline, index, update)


def _mapping_b(timeline, index):
    def update(_state, ptype):
        if ptype == "HH":
            return 1
        if ptype == "LL":
            return 0
        return None
    return _timeline_to_position(timeline, index, update)


def _mapping_c(timeline, index):
    def update(state, ptype):
        if ptype == "LL":
            state["recent_LL"] = True
            state["recent_HH"] = False
            return None
        if ptype == "HH":
            state["recent_HH"] = True
            state["recent_LL"] = False
            return None
        if ptype == "HL" and state["recent_LL"] and state["pos"] == 0:
            state["recent_LL"] = False
            return 1
        if ptype == "LH" and state["recent_HH"] and state["pos"] == 1:
            state["recent_HH"] = False
            return 0
        return None
    return _timeline_to_position(timeline, index, update)


_MAPPERS = {"A": _mapping_a, "B": _mapping_b, "C": _mapping_c}


# ---------------------------------------------------------------------------
# Public contract: precompute + apply_thresholds
# ---------------------------------------------------------------------------


def precompute(
    close: pd.Series,
    volume: pd.Series,
    method: str,
    method_param: float | int,
    mapping: str,
    n_compare: int = 1,
) -> dict[str, Any]:
    """Build causal OBV + pivot timeline + pre-stop 0/1 position series.

    Returns:
        obv:        pd.Series — cumulative OBV on the close index
        timeline:   list[(act_day, ptype)] — causal pivot events
        position:   pd.Series of 0/1 on close.index (before stops, ffilled)
    """
    if method not in PIVOT_METHODS:
        raise ValueError(f"method must be one of {PIVOT_METHODS}, got {method!r}")
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    obv = compute_obv(close, volume)
    timeline = causal_timeline(obv, method, method_param, n_compare=n_compare)
    position = _MAPPERS[mapping](timeline, close.index)
    return {"obv": obv, "timeline": timeline, "position": position}


def apply_thresholds(
    pre: dict[str, Any],
    stop_type: str = "none",
    stop_param: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(entries, exits)`` boolean Series on the close index.

    Stops are applied by the S2 walker (needs high/low/close + ATR), not here.
    This helper just returns entry/exit edges of the raw pre-stop position so
    the lib.Pipeline path can still consume the signal without stops.
    """
    pos = pre["position"].astype(int)
    prev = pos.shift(1).fillna(0).astype(int)
    entries = (pos == 1) & (prev == 0)
    exits = (pos == 0) & (prev == 1)
    return entries, exits
