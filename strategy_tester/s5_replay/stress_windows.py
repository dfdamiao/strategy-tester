"""Named stress / interesting-time windows.

Port of `pyfolio.interesting_periods.PERIODS` (20 historical events through
~2016) + 5 modern additions (COVID, 2022 rate shock, SVB, Yen unwind 2024,
2024 Trump-trade rally) so reports generated 2026+ have post-pyfolio context.

Each window is a (start_date, end_date) tuple. Use `slice_returns` /
`window_stats` to compute per-window metrics from a portfolio + benchmark
equity series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Order: chronological. Tuples are pd.Timestamp-friendly strings.
WINDOWS: dict[str, tuple[str, str]] = {
    # pyfolio.interesting_periods (canonical 20)
    "US_Housing_2003":           ("2003-01-08", "2003-02-08"),
    "Aug07_QuantQuake":          ("2007-08-01", "2007-09-01"),
    "Mar08_BearStearns":         ("2008-03-01", "2008-04-01"),
    "Sep08_Lehman":              ("2008-09-01", "2008-10-01"),
    "Lehman":                    ("2008-08-01", "2008-10-01"),
    "2009Q1":                    ("2009-01-01", "2009-03-01"),
    "2009Q2":                    ("2009-03-01", "2009-06-01"),
    "Flash_Crash_May2010":       ("2010-05-05", "2010-05-10"),
    "Fukushima_Mar2011":         ("2011-03-16", "2011-04-16"),
    "US_Downgrade_Aug2011":      ("2011-08-05", "2011-09-05"),
    "EZB_Rate_Sep2012":          ("2012-09-10", "2012-10-10"),
    "Apr14":                     ("2014-04-01", "2014-05-01"),
    "Oct14_Bund_Tantrum":        ("2014-10-01", "2014-11-01"),
    "Fall2015_Aug_Tantrum":      ("2015-08-15", "2015-09-30"),
    "Dotcom_2000":               ("2000-03-10", "2000-09-10"),
    "9_11":                      ("2001-09-11", "2001-10-11"),
    "Low_Vol_Bull_05_07":        ("2005-01-01", "2007-08-01"),
    "GFC_Crash":                 ("2007-08-01", "2009-04-01"),
    "Recovery_09_13":            ("2009-04-01", "2013-01-01"),
    "New_Normal":                ("2013-01-01", "2020-01-01"),
    # 5 modern additions (post-pyfolio)
    "COVID_Crash_2020":          ("2020-02-19", "2020-03-23"),
    "COVID_Recovery_2020":       ("2020-03-24", "2020-12-31"),
    "Rate_Shock_2022":           ("2022-01-03", "2022-10-13"),
    "SVB_Mar2023":               ("2023-03-08", "2023-03-15"),
    "Yen_Unwind_Aug2024":        ("2024-07-31", "2024-08-09"),
    "Trump_Trade_Q4_2024":       ("2024-11-05", "2024-12-31"),
}


def slice_equity(
    eq: pd.Series, start: str, end: str,
) -> pd.Series:
    """Return eq sliced to [start, end]. May be empty if window pre-dates eq."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return eq.loc[(eq.index >= start_ts) & (eq.index <= end_ts)]


def window_stats(
    eq: pd.Series, bench_eq: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-window cum-return / Sharpe / MaxDD vs benchmark."""
    rows = []
    for name, (start, end) in WINDOWS.items():
        s = slice_equity(eq, start, end)
        if len(s) < 5:
            continue
        cum_ret = float(s.iloc[-1] / s.iloc[0] - 1)
        rets = s.pct_change().dropna()
        sd = float(rets.std(ddof=1))
        sr = float(rets.mean() / sd * np.sqrt(252)) if sd > 1e-12 else np.nan
        peak = s.cummax()
        max_dd = float((s / peak - 1).min())
        bench_cum = np.nan
        excess = np.nan
        if bench_eq is not None:
            bs = slice_equity(bench_eq, start, end)
            if len(bs) >= 5:
                bench_cum = float(bs.iloc[-1] / bs.iloc[0] - 1)
                excess = cum_ret - bench_cum
        rows.append({
            "window": name,
            "start": start,
            "end": end,
            "n_bars": int(len(s)),
            "cum_return": cum_ret,
            "bench_return": bench_cum,
            "excess_return": excess,
            "sharpe": sr,
            "max_dd": max_dd,
        })
    return pd.DataFrame(rows)
