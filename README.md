# strategy-tester

A staged research pipeline for testing systematic trading strategies. It takes
a universe of candidate pairs, screens them, optimizes an entry/exit signal,
validates out-of-sample, significance-tests the survivors, and turns the
surviving cohort into a weighted portfolio. Backtests run on the stock
[`vectorbt`](https://github.com/polakowo/vectorbt) package; price data comes
from Yahoo Finance (keyless).

The point of a pipeline like this is to **reject noise**. Most candidates
should die at one of the gates; a shrinking funnel is the healthy outcome.

## The pipeline

Each stage is a gate that shrinks the candidate set:

| Stage | Name | What it does |
|---|---|---|
| **S1** | screen | Mean-reversion / cointegration screen (halflife, Hurst, ADF, efficiency ratio, Johansen, ...) |
| **S2** | optimize | Grid / Optuna search of the entry-exit signal per surviving pair, with an IS/OOS gate |
| **S3** | validate | Out-of-sample validation: walk-forward (Pardo), CPCV (Lopez de Prado), bootstrap, Monte Carlo |
| **S4** | significance | Significance tests on the survivors: t-test, PSR, DSR, permutation, Carver 2-sigma |
| **S5** | portfolio | Turn the cohort into weights: equal, inverse-vol, Sharpe-weighted, HRP, Kelly, ... |

Every stage's methods are published techniques; each module's docstring cites
its source (Chan, Pardo, Lopez de Prado, Bailey & Lopez de Prado, Carver,
Kaufman, and others).

## Install

```bash
git clone https://github.com/dfdamiao/strategy-tester.git
cd strategy-tester
pip install -e .
# or: pip install -r requirements.txt
```

Python >= 3.11. The backtest engine is tested with `vectorbt` 0.28.4.

## Quick start

**One backtest** (touches only the engine, fastest sanity check):

```bash
python -m examples.single_backtest
```

```
SPY/QQQ mean-reversion backtest (stock vectorbt)
  Sharpe   : 0.50
  CAGR     : 1.84%
  Max DD   : -4.49%
  Trades   : 2
  Win rate : 50.0%
```

**The full S1->S5 pipeline** on a basket of real ETFs:

```bash
python -m examples.run_pipeline
```

```
Pipeline: WFA-expanding demo
  S1 screened : 105 candidate pairs
  S2 optimized: 35 passed the signal gate
  S3 validated: 24 walk-forward survivors
  S4 cohort   : 7 significant pairs (of 23 tested)

  S5 [equal_weight]: 7 weighted positions
    portfolio Sharpe : 0.52
    portfolio CAGR   : 8.16%
    max drawdown     : -41.42%
      XLF/SLV       14.3%
      ...
```

(The thresholds in the examples are deliberately loose so a cohort survives on
a small universe. The defaults in `strategy_tester/config.py` are stricter.)

## Driving it yourself

```python
import itertools, yfinance as yf
from strategy_tester import Pipeline, list_methods

tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
prices = yf.download(tickers, period="10y", auto_adjust=False,
                     progress=False)["Close"].dropna(how="all").ffill()
pairs = [{"pair": f"{a}/{b}", "numerator": a, "denominator": b}
         for a, b in itertools.combinations(tickers, 2)]

pipe = Pipeline(
    s1="chan_halflife", s2_signal="zscore_robust_mad", s2_optim="grid_search",
    s3="wfa_expanding", s4="t_test", s5="equal_weight",
)
result = pipe.run(prices, pairs, config={"n_folds": 5, "t_stat_threshold": 1.0})

print(result.stages["s4"]["result"])   # the significant cohort
print(result.final)                     # {s5_method: portfolio dict}
```

`list_methods()` returns every available option per stage. Ready-made stage
combinations ("presets") live in `strategy_tester/presets.py`:

```python
from strategy_tester import Pipeline, PRESETS
pipe = Pipeline(**PRESETS["P-1"])
```

Stop early with `pipe.run(..., stop_after="s2")` to inspect an intermediate
stage. Use `report=True, output_dir=...` to write an HTML report.

## Just the backtest engine

If you only want to backtest a single signal, skip the pipeline:

```python
from strategy_tester.backtest.vbt_runner import backtest_vbt_fold

res = backtest_vbt_fold(
    num_prices=prices["SPY"],
    ratio=(prices["SPY"] / prices["QQQ"]),
    window=20, entry_thresh=-2.0, exit_thresh=0.5, fees=0.001,
)
# {'sharpe', 'cagr', 'max_dd', 'n_trades', 'hit_rate', 'returns'}
```

There is also a pure-Numba backtest path (`backtest_numba_fold`) that needs no
vectorbt at all, used for fast grid sweeps.

## Bringing your own data

The pipeline takes a plain pandas DataFrame of close prices (DatetimeIndex x
tickers) plus a list of pair dicts, so any source works. A yfinance loader is
bundled (`strategy_tester.s0_features.bar_loader.load_universe_bars`,
`strategy_tester.data.get_prices`), but you can pass your own frame directly.

## Project layout

```
strategy_tester/
  s0_features/   feature engineering + the yfinance bar loader
  s1_screening/  mean-reversion / cointegration screens
  s2_signal/     signal methods (z-score, EMA/MA cross, momentum, RSI, ...)
  s2_optimize/   grid / Optuna / Numba grid search
  s3_validation/ walk-forward, CPCV, bootstrap, Monte Carlo
  s4_significance/ t-test, PSR, DSR, permutation, Carver
  s5_portfolio/  weighting schemes
  s5_replay/     cash-aware portfolio simulator
  backtest/      vectorbt + pure-Numba backtest engine
  pipeline.py    the Pipeline orchestrator
  registry.py    method discovery (list_methods / get_method)
  presets.py     ready-made stage combinations
examples/        runnable single-backtest + full-pipeline demos
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT (see [LICENSE](LICENSE)). Backtests run on the stock `vectorbt` package;
attribution and a note on the author's private fork are in [NOTICE](NOTICE).
