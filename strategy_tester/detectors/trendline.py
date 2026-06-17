"""Trendline detection on OHLC series.

Ported from ``ratio_breakout_strategy/trendline_detector.py``. Kernel unchanged;
multiprocessing switched to ``mp.get_context("fork") + Pool(initializer=...)``
to match the ``grid_ma`` convention used by other strategies in this library.

Algorithm (brute force O(n^2), Numba-JIT'd):
- For each pair (i, j) of bar indices with j > i, fit a line through highs
  (resistance) or lows (support).
- Count points touching the line within ``max_error`` tolerance; reject if
  any bar breaches past the breakout threshold.
- Score surviving lines by ``-log(avg_error) + num_points * log(2.5)``.

Source files preserved for reference:
- ratio_breakout_strategy/trendline_detector.py
- ratio_breakout_strategy/README.md (documents the overflow + NaN fixes)
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd


@dataclass
class TrendlineResult:
    """One detected trendline."""

    start_idx: int
    end_idx: int
    slope: float
    intercept: float
    num_points: int
    score: float
    start_price: float
    end_price: float
    slope_angle_degrees: float
    type: str  # "resistance" or "support"


# ---------------------------------------------------------------------------
# Numba JIT kernel (unchanged from ratio_breakout_strategy, lines 149-260)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _process_chunk_numba(
    chunk_start,
    chunk_end,
    high_prices,
    low_prices,
    open_prices,
    close_prices,
    min_points,
    max_error,
    breakout_tolerance,
    min_period_days,
    max_period_days,
    is_resistance,
    max_results,
):
    """O(n^2) trendline kernel. Returns (K, 8) float64 array.

    Row layout: [start_idx, end_idx, slope, intercept, num_points, score,
    start_price, end_price]. If K == max_results, caller should retry with
    a larger buffer.
    """
    n = len(high_prices)
    results = np.empty((max_results, 8), dtype=np.float64)
    result_count = 0
    points_buf = np.empty(n, dtype=np.int64)

    for i in range(chunk_start, min(chunk_end, n - 1)):
        for j in range(i + 1, n):
            period_days = j - i
            if min_period_days > 0 and period_days < min_period_days:
                continue
            if max_period_days > 0 and period_days > max_period_days:
                continue

            if is_resistance:
                price_i = high_prices[i]
                price_j = high_prices[j]
            else:
                price_i = low_prices[i]
                price_j = low_prices[j]

            slope = (price_j - price_i) / (j - i)

            if is_resistance and slope >= 0.0:
                continue
            if not is_resistance and slope <= 0.0:
                continue

            intercept = price_i - slope * i

            n_points = 0
            valid = True

            for k in range(i, j + 1):
                expected_price = slope * k + intercept
                high_price = high_prices[k]
                low_price = low_prices[k]
                open_price = open_prices[k]
                close_price = close_prices[k]

                if is_resistance:
                    breach = high_price - expected_price - breakout_tolerance
                    if breach > max_error:
                        valid = False
                        break
                    tolerance_zone = expected_price * (1.0 - breakout_tolerance)
                    if (
                        abs(high_price - expected_price) <= max_error
                        or abs(open_price - expected_price) <= max_error
                        or abs(close_price - expected_price) <= max_error
                    ):
                        points_buf[n_points] = k
                        n_points += 1
                    elif high_price >= tolerance_zone:
                        points_buf[n_points] = k
                        n_points += 1
                else:
                    breach = expected_price - low_price - breakout_tolerance
                    if breach > max_error:
                        valid = False
                        break
                    tolerance_zone = expected_price * (1.0 + breakout_tolerance)
                    if (
                        abs(low_price - expected_price) <= max_error
                        or abs(open_price - expected_price) <= max_error
                        or abs(close_price - expected_price) <= max_error
                    ):
                        points_buf[n_points] = k
                        n_points += 1
                    elif low_price <= tolerance_zone:
                        points_buf[n_points] = k
                        n_points += 1

            if not valid or n_points < min_points:
                continue

            error_sum = 0.0
            for p in range(n_points):
                k = points_buf[p]
                if is_resistance:
                    error_sum += abs(high_prices[k] - (slope * k + intercept))
                else:
                    error_sum += abs(low_prices[k] - (slope * k + intercept))
            avg_error = error_sum / n_points
            score = -math.log(avg_error + 0.001) + n_points * math.log(2.5)

            if result_count < max_results:
                results[result_count, 0] = i
                results[result_count, 1] = j
                results[result_count, 2] = slope
                results[result_count, 3] = intercept
                results[result_count, 4] = n_points
                results[result_count, 5] = score
                results[result_count, 6] = price_i
                results[result_count, 7] = price_j
                result_count += 1

    return results[:result_count]


# ---------------------------------------------------------------------------
# Chunk workers (must be module-level for fork Pool)
# ---------------------------------------------------------------------------


def _worker_init():
    """Silence logs in forked workers."""
    import warnings

    warnings.filterwarnings("ignore")


def _process_chunk(args):
    """Wrapper that calls the Numba kernel and boxes results into dataclasses."""
    (
        chunk_start,
        chunk_end,
        high_prices,
        low_prices,
        open_prices,
        close_prices,
        params,
        mode,
    ) = args
    (
        min_points,
        max_error,
        breakout_tolerance,
        min_period_days,
        max_period_days,
    ) = params

    is_resistance = mode == "resistance"
    mpd = max_period_days if max_period_days is not None else -1
    mpd_min = min_period_days if min_period_days is not None else -1

    buf_size = 100_000
    while True:
        raw = _process_chunk_numba(
            chunk_start,
            chunk_end,
            high_prices,
            low_prices,
            open_prices,
            close_prices,
            min_points,
            max_error,
            breakout_tolerance,
            mpd_min,
            mpd,
            is_resistance,
            buf_size,
        )
        if raw.shape[0] < buf_size:
            break
        buf_size *= 2

    out: list[TrendlineResult] = []
    for row_idx in range(raw.shape[0]):
        row = raw[row_idx]
        out.append(
            TrendlineResult(
                start_idx=int(row[0]),
                end_idx=int(row[1]),
                slope=float(row[2]),
                intercept=float(row[3]),
                num_points=int(row[4]),
                score=float(row[5]),
                start_price=float(row[6]),
                end_price=float(row[7]),
                slope_angle_degrees=float(np.degrees(np.arctan(row[2]))),
                type=mode,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class HybridTrendDetector:
    """Detect resistance / support trendlines from OHLC data.

    Parameters
    ----------
    min_points : number of bars that must touch the line (default 4)
    max_error_pct : tolerance as % of avg candle range (default 2.0)
    breakout_tolerance_pct : extra slack before line is "broken" (default 0.0)
    min_period_days : minimum line length in bars (default 21)
    max_period_days : optional maximum line length in bars
    n_cores : worker count; default ``cpu_count() - 2``
    """

    def __init__(
        self,
        min_points: int = 4,
        max_error_pct: float = 2.0,
        breakout_tolerance_pct: float = 0.0,
        min_period_days: int = 21,
        max_period_days: int | None = None,
        n_cores: int | None = None,
    ) -> None:
        self.min_points = min_points
        self.max_error_pct = max_error_pct
        self.max_error = max_error_pct / 100.0
        self.breakout_tolerance_pct = breakout_tolerance_pct
        self.breakout_tolerance = breakout_tolerance_pct / 100.0
        self.min_period_days = min_period_days
        self.max_period_days = max_period_days
        self.n_cores = n_cores if n_cores else max(1, mp.cpu_count() - 2)

    def _avg_candle_range(self, df: pd.DataFrame) -> float:
        range_series = df["High"] - df["Low"]
        avg = range_series.mean()
        if np.isnan(avg) or avg < 0.001:
            avg = df["Close"].mean() * 0.02
        if np.isnan(avg) or avg < 0.001:
            avg = 0.001
        return float(avg)

    def detect(
        self, df: pd.DataFrame, mode: str = "resistance",
    ) -> list[TrendlineResult]:
        """Detect lines of the given ``mode`` on ``df`` (needs High/Low/Open/Close)."""
        if mode not in ("resistance", "support"):
            raise ValueError("mode must be 'resistance' or 'support'")

        df = pd.DataFrame(df[["High", "Low", "Open", "Close"]]).copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if len(df) < 100:
            return []

        high_prices = df["High"].to_numpy(dtype=np.float64)
        low_prices = df["Low"].to_numpy(dtype=np.float64)
        open_prices = df["Open"].to_numpy(dtype=np.float64)
        close_prices = df["Close"].to_numpy(dtype=np.float64)
        n = len(df)

        avg_range = self._avg_candle_range(df)
        if np.isnan(avg_range):
            return []
        max_error = self.max_error * avg_range
        breakout_tolerance = self.breakout_tolerance * avg_range

        params = (
            self.min_points,
            max_error,
            breakout_tolerance,
            self.min_period_days,
            self.max_period_days,
        )

        chunk_size = max(1, n // self.n_cores)
        chunks: list[tuple[int, int]] = []
        for i in range(self.n_cores):
            chunk_start = i * chunk_size
            chunk_end = n if i == self.n_cores - 1 else (i + 1) * chunk_size
            chunks.append((chunk_start, chunk_end))

        chunk_args = [
            (
                cs,
                ce,
                high_prices,
                low_prices,
                open_prices,
                close_prices,
                params,
                mode,
            )
            for cs, ce in chunks
        ]

        if self.n_cores == 1:
            return _sort_by_score(_process_chunk(chunk_args[0]))

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=self.n_cores, initializer=_worker_init) as pool:
            results = pool.map(_process_chunk, chunk_args)

        all_lines: list[TrendlineResult] = []
        for r in results:
            if r:
                all_lines.extend(r)
        return _sort_by_score(all_lines)


def _sort_by_score(lines: list[TrendlineResult]) -> list[TrendlineResult]:
    lines.sort(key=lambda x: x.score, reverse=True)
    return lines


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def detect_resistance(
    df: pd.DataFrame,
    *,
    min_points: int = 4,
    max_error_pct: float = 2.0,
    min_period_days: int = 21,
    n_cores: int | None = 1,
) -> list[TrendlineResult]:
    """Resistance lines on ``df`` (needs High/Low/Open/Close).

    Default ``n_cores=1`` — parallelism is usually better done across pairs by
    the caller (e.g. grid_breakout), not across chunks of a single pair.
    """
    det = HybridTrendDetector(
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=n_cores,
    )
    return det.detect(df, mode="resistance")


def detect_support(
    df: pd.DataFrame,
    *,
    min_points: int = 4,
    max_error_pct: float = 2.0,
    min_period_days: int = 21,
    n_cores: int | None = 1,
) -> list[TrendlineResult]:
    """Support lines on ``df`` (needs High/Low/Open/Close)."""
    det = HybridTrendDetector(
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=n_cores,
    )
    return det.detect(df, mode="support")
