"""Persistence helpers for the in-memory ReplayState pieces that aren't
saved by the existing pipeline.py (daily_snapshot, trades, failed_entries).

This lets a strategy pipeline.py persist enough state to re-render the
v2 mega-report (cash mobilization 4-panel + Trades tab) without re-running
the backtest. Optional — pipelines opt in by calling these helpers
alongside the equity/metrics parquet emissions they already do.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_tester.s5_replay.charts.trades import trades_to_dataframe


def save_daily_snapshot(snapshot: list[dict], path: Path) -> None:
    """Save the per-bar (date, cash, position_value, netliq, peak_nav, dd,
    n_positions) snapshot to parquet."""
    if not snapshot:
        return
    df = pd.DataFrame(snapshot)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.to_parquet(path)


def load_daily_snapshot(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    df = df.reset_index()
    return df.to_dict("records")


def save_trades_log(trades: list, path: Path) -> None:
    """Save closed trades to parquet (one row per round trip)."""
    df = trades_to_dataframe(trades)
    if df.empty:
        return
    df.to_parquet(path, index=False)


def load_trades_log(path: Path) -> pd.DataFrame:
    """Read trades parquet → DataFrame in the canonical schema. Empty DF
    if missing."""
    if not path.exists():
        return trades_to_dataframe([])
    return pd.read_parquet(path)


def save_failed_entries(failed: list, path: Path) -> None:
    """Save FailedEntry list (ticker, date, target_dollars, sized_dollars,
    cash_available, reason)."""
    if not failed:
        return
    df = pd.DataFrame([
        {
            "ticker": f.ticker, "date": pd.Timestamp(f.date),
            "target_dollars": f.target_dollars,
            "sized_dollars": f.sized_dollars,
            "cash_available": f.cash_available,
            "reason": f.reason,
        }
        for f in failed
    ])
    df.to_parquet(path, index=False)


def load_failed_entries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["ticker", "date", "target_dollars",
                     "sized_dollars", "cash_available", "reason"],
        )
    return pd.read_parquet(path)


def save_scheme_state_bundle(
    state, out_dir: Path, scheme: str,
) -> dict[str, Path]:
    """Persist all three pieces of a single scheme's ReplayState to disk
    under `<out_dir>/<prefix>_{snapshots,trades,failed_entries}.parquet`.

    Returns the {kind: path} dict so the caller can register them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "snapshots": out_dir / f"{scheme}_snapshots.parquet",
        "trades": out_dir / f"{scheme}_trades.parquet",
        "failed_entries": out_dir / f"{scheme}_failed_entries.parquet",
    }
    save_daily_snapshot(state.daily_snapshot, paths["snapshots"])
    save_trades_log(state.trades, paths["trades"])
    save_failed_entries(state.failed_entries, paths["failed_entries"])
    return paths
