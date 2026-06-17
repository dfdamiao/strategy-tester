"""End-to-end example: run the full S1->S5 pipeline on real ETF data.

Stages (each one is a gate that shrinks the candidate set):
    S1  screen     candidate pairs (mean-reversion / cointegration)
    S2  optimize   the entry/exit signal per surviving pair
    S3  validate   out-of-sample via walk-forward (multiple folds)
    S4  significance test the survivors (t-test on fold Sharpes)
    S5  portfolio  turn the surviving cohort into weights

A healthy validation pipeline rejects most candidates; a shrinking funnel is
the point, not a bug. Tighten the thresholds below for stricter research.

Run from the repo root:
    python -m examples.run_pipeline
"""
from __future__ import annotations

import itertools

import yfinance as yf

from strategy_tester import Pipeline


def main() -> None:
    tickers = [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "XLE", "XLF",
        "HYG", "LQD", "DIA", "VWO", "SLV", "USO", "XLK",
    ]
    raw = yf.download(
        tickers, period="12y", auto_adjust=False, progress=False
    )
    prices = raw["Close"].dropna(how="all").ffill()

    # Candidate universe = every 2-ticker ratio.
    pairs = [
        {"pair": f"{a}/{b}", "numerator": a, "denominator": b}
        for a, b in itertools.combinations(tickers, 2)
    ]

    # Build the pipeline stage by stage. We use a MULTI-FOLD walk-forward
    # (wfa_expanding) for S3 so the S4 t-test has >=2 fold Sharpes to test.
    # list_methods() (from strategy_tester) shows every available stage option;
    # ready-made combinations live in strategy_tester/presets.py.
    pipe = Pipeline(
        s1="chan_halflife",
        s2_signal="zscore_robust_mad",
        s2_optim="grid_search",
        s3="wfa_expanding",
        s4="t_test",
        s5="equal_weight",
        name="WFA-expanding demo",
    )

    # Illustrative config. Defaults in strategy_tester/config.py are stricter.
    config = {
        "entry_grid": [-2.5, -2.0, -1.5, -1.0],
        "exit_grid": [0.0, 0.5, 1.0],
        "stop_grid": [0.0],
        "slope_grid": [0.0],
        "adf_pvalue_threshold": 0.40,
        "min_is_sharpe": 0.0,
        "min_is_trades": 3,
        "min_oos_trades": 1,
        "n_folds": 5,
        "t_stat_threshold": 0.5,
    }

    result = pipe.run(prices, pairs, config, report=False)

    print(f"\nPipeline: {result.name}")
    print(f"  S1 screened : {len(result.stages['s1']['result'])} candidate pairs")
    print(f"  S2 optimized: {len(result.stages['s2']['result'])} passed the signal gate")
    print(f"  S3 validated: {len(result.stages['s3']['result'])} walk-forward survivors")
    s4 = result.stages["s4"]["result"]
    n_sig = int(s4["passed"].sum()) if "passed" in s4 else 0
    print(f"  S4 cohort   : {n_sig} significant pairs (of {len(s4)} tested)")

    # result.final is {s5_method_name: portfolio_result_dict}.
    for method, port in (result.final or {}).items():
        n = port.get("n_pairs", 0)
        print(f"\n  S5 [{method}]: {n} weighted positions")
        if n:
            print(f"    portfolio Sharpe : {port.get('sharpe', 0):.2f}")
            print(f"    portfolio CAGR   : {port.get('cagr', 0) * 100:.2f}%")
            print(f"    max drawdown     : {port.get('max_dd', 0) * 100:.2f}%")
            for pair, w in port.get("weights", {}).items():
                print(f"      {pair:<12} {w:6.1%}")
        else:
            print("    empty cohort — loosen the config or widen the universe")


if __name__ == "__main__":
    main()
