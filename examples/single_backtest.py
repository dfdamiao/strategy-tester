"""Minimal example: backtest one mean-reversion signal on a real price ratio.

Run from the repo root:
    python -m examples.single_backtest

This touches only the backtest engine (strategy_tester.backtest) and stock
vectorbt, so it is the quickest way to confirm your install works.
"""
from __future__ import annotations

import yfinance as yf

from strategy_tester.backtest.vbt_runner import backtest_vbt_fold


def main() -> None:
    # 1) Real prices, keyless, from Yahoo Finance.
    raw = yf.download(
        ["SPY", "QQQ"], period="3y", auto_adjust=False, progress=False
    )
    close = raw["Close"].dropna()
    num, den = close["SPY"], close["QQQ"]

    # 2) Signal = robust z-score of the SPY/QQQ price ratio. Go long the ratio
    #    when it is cheap (z <= entry_thresh), exit when it reverts up
    #    (z >= exit_thresh). The engine computes the z-score from the ratio.
    ratio = (num / den).rename("SPY/QQQ")

    # 3) One vectorbt backtest -> metrics as a plain dict.
    res = backtest_vbt_fold(
        num_prices=num,
        ratio=ratio,
        window=20,
        entry_thresh=-2.0,
        exit_thresh=0.5,
        fees=0.001,
    )

    print("SPY/QQQ mean-reversion backtest (stock vectorbt)")
    print(f"  Sharpe   : {res['sharpe']:.2f}")
    print(f"  CAGR     : {res['cagr']:.2f}%")
    print(f"  Max DD   : {res['max_dd']:.2f}%")
    print(f"  Trades   : {res['n_trades']}")
    print(f"  Win rate : {res['hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
