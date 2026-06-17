"""Deflated Sharpe Ratio. Bailey & López de Prado (2014).

DSR ≡ PSR(SR* = E[max(SR_i)_{i=1..N}]) — corrects PSR for the multiple-testing
bias introduced when the reported SR is the winner of N parameter trials.

Mathematical reference: Bailey & López de Prado, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
Journal of Portfolio Management, 2014, Eq. 5 (E[max]) and Eq. 10 (DSR).

CRITICAL UNIT CONVENTION:
    E[max(SR) | N] is expressed in SR units, NOT standard-normal units.
    The conversion is:
        E[max(SR)] = sqrt(Var(SR)) × E[max(N standard normals)]

    The standard-normal expected max (Gumbel approximation) is:
        z_max(N) = (1-γ_e)·Φ⁻¹(1-1/N) + γ_e·Φ⁻¹(1-1/(N·e))
        where γ_e = 0.5772156649 (Euler-Mascheroni constant)

    Subtracting z_max(N) from SR_obs without scaling by sqrt(Var(SR))
    is a UNIT MISMATCH — z_max is dimensionless, SR is in SR units.
    See tests/test_dsr_math.py::test_dimensional_consistency.

    A previous version of this module had exactly that bug (pre-2026-05-15)
    and produced systematically-deflated DSR values near zero. Fixed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from strategy_tester.backtest.metrics import psr_stat
from strategy_tester.registry import register_stage

EULER_MASCHERONI: float = 0.5772156649


def expected_max_sharpe(n_trials: int | float, sr_variance: float) -> float:
    """E[max(SR_i)] under H0 of zero true SR, Bailey-LdP 2014 Eq.5.

    Returns the expected maximum SR in the SAME unit-system as sqrt(sr_variance).
    Pass annualized sr_variance to get annualized E[max]; pass per-period to get
    per-period. Caller must keep annualization consistent across SR_obs and
    the returned floor.

    Parameters
    ----------
    n_trials : int | float
        Number of trials (independent search candidates). Float allowed for
        effective_N from correlation correction.
    sr_variance : float
        Variance of SR estimates across the n_trials candidates.

    Returns
    -------
    float
        Expected maximum SR. Zero if n_trials < 2 or variance non-positive.
    """
    if n_trials < 2 or sr_variance <= 0 or not np.isfinite(sr_variance):
        return 0.0
    sqrt_var = float(np.sqrt(sr_variance))
    z_n = float(norm.ppf(1.0 - 1.0 / n_trials))
    z_ne = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    z_max = (1 - EULER_MASCHERONI) * z_n + EULER_MASCHERONI * z_ne
    return sqrt_var * z_max


def effective_n_from_correlation(rho_bar: float, n_trials: int) -> float:
    """LdP AFML Ch.14: effective N for correlated trials = N^(1 − ρ̄).

    Bounds rho_bar to [0, 0.999] for numerical safety. Floor at 1.001 so
    log(N_eff) > 0.

    Parameters
    ----------
    rho_bar : float
        Mean pairwise correlation of trial return series in [0, 1).
    n_trials : int
        Raw number of trials.

    Returns
    -------
    float
        Effective independent trial count. ≥ 1.001.
    """
    rho_clipped = max(0.0, min(float(rho_bar), 0.999))
    return max(float(n_trials) ** (1.0 - rho_clipped), 1.001)


def deflated_sharpe(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    rho_bar: float | None = None,
) -> float:
    """Deflated Sharpe Ratio probability. Bailey-LdP 2014 Eq.10.

    DSR = PSR(SR* = E[max(SR) | N_effective])

    Parameters
    ----------
    sharpe : float
        Observed Sharpe ratio. Same annualization as sqrt(sr_variance).
    n_obs : int
        Number of underlying return observations (or fold count if working
        with cross-fold SR distribution).
    n_trials : int
        Raw trial count from S2 grid search.
    sr_variance : float
        Variance of SR estimates across the n_trials trials. Must be in the
        same annualization as `sharpe`.
    skew, kurtosis : float
        Sample skewness and kurtosis. Pass (0, 3) for normal-returns assumption.
    rho_bar : float | None
        Mean pairwise correlation of trial returns. If provided, effective
        n_trials = n_trials^(1 − ρ̄) per LdP AFML Ch.14. If None, raw n_trials
        is used (Bailey-LdP strict, which assumes independent trials).

    Returns
    -------
    float
        DSR probability in [0, 1]. Higher = more confident the observed SR is
        not the result of multiple-testing selection bias.
    """
    if rho_bar is None and n_trials > 1:
        import warnings

        warnings.warn(
            "deflated_sharpe called with rho_bar=None and n_trials > 1 "
            f"({n_trials}). Per docs/rules.md §3g, per_asset "
            "DSR MUST pass an empirical rho_bar (mean pairwise correlation "
            "of trial return series). raw-N assumes independent trials and "
            "systematically under-counts the DSR cohort when trials share a "
            "signal kernel (typical empirical rho_bar 0.65-0.85). "
            "See lib/s5_replay/compute_rho_bar.py for the canonical helper.",
            DeprecationWarning,
            stacklevel=2,
        )

    n_eff = (
        effective_n_from_correlation(rho_bar, n_trials)
        if rho_bar is not None else float(n_trials)
    )
    sr_benchmark = expected_max_sharpe(n_eff, sr_variance)
    return psr_stat(sharpe, n_obs, skew, kurtosis, sr_benchmark)


@register_stage("s4")
def dsr(s3_result: pd.DataFrame, **config) -> pd.DataFrame:
    """DSR S4 gate. Bailey-LdP 2014.

    Required config
    ---------------
    n_trials : int
        Number of S2 parameter combos tested. Use the effective count if known
        (or set ``rho_bar`` to derive it automatically).

    Optional config
    ---------------
    sr_variance : float
        Variance of SR across the n_trials combos. If absent, falls back to
        ``std_test_sharpe**2`` from each row (cross-fold dispersion).
    rho_bar : float
        Mean pairwise correlation of S2 trial returns ∈ [0, 1). Effective N =
        N^(1 − ρ̄). Defaults to ``None`` (use raw n_trials).
    dsr_alpha : float
        Significance level (default 0.05 → DSR > 0.95 to pass).
    skew, kurtosis : float
        Defaults (0, 3) — normal-returns assumption. Pass observed values
        per row via ``skew`` / ``kurtosis`` columns for non-normal returns.

    Per-row column overrides (if present)
    -------------------------------------
    ``sr_variance``, ``skew``, ``kurtosis`` take precedence over config defaults.
    """
    n_trials = config.get("n_trials")
    if n_trials is None:
        raise KeyError(
            "n_trials required for DSR — set config['n_trials'] "
            "to the number of parameter combos tested in S2"
        )
    alpha = config.get("dsr_alpha", 0.05)
    rho_bar_cfg = config.get("rho_bar", None)
    skew_cfg = config.get("skew", 0.0)
    kurt_cfg = config.get("kurtosis", 3.0)

    def _safe_float(val: object, default: float) -> float:
        """Coerce row cell to float, falling back to default on None/NaN."""
        if val is None:
            return default
        try:
            f = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return default if np.isnan(f) else f

    rows = []
    for _, row in s3_result[s3_result["passed"]].iterrows():
        sr = _safe_float(row["mean_test_sharpe"], 0.0)

        # Paper-correct n_obs is bar count T. Fall back to fold count with
        # a deprecation warning when upstream S3 lacks the column.
        # See METHODOLOGY_DECISIONS.md §1.
        n_bars = _safe_float(row.get("n_oos_bars"), float("nan"))
        if np.isnan(n_bars) or n_bars <= 0:
            import warnings

            warnings.warn(
                "lib.s4_significance.dsr: row missing 'n_oos_bars'; "
                "falling back to 'n_test_periods' (fold count). Re-run "
                "S3 to populate the bar-count column. "
                "See METHODOLOGY_DECISIONS.md §1.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_bars = _safe_float(row.get("n_test_periods"), 0.0)
        n_obs = int(n_bars)

        sr_var_row = _safe_float(row.get("sr_variance"), float("nan"))
        if np.isnan(sr_var_row):
            std_sr = _safe_float(row.get("std_test_sharpe"), 0.5)
            if std_sr == 0:
                std_sr = 0.5
            sr_var_row = std_sr ** 2

        skew_row = _safe_float(row.get("skew"), skew_cfg)
        kurt_row = _safe_float(row.get("kurtosis"), kurt_cfg)

        p = deflated_sharpe(
            sharpe=sr,
            n_obs=max(n_obs, 2),
            n_trials=int(n_trials),
            sr_variance=sr_var_row,
            skew=skew_row,
            kurtosis=kurt_row,
            rho_bar=rho_bar_cfg,
        )
        passed = p > (1 - alpha)
        rows.append({
            "pair": row["pair"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "passed": passed,
            "tier": "TOP_TIER" if passed else "REJECT",
            "dsr_stat": round(p, 4),
            "dsr_passed": passed,
            "sig_method": "dsr",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pair", "numerator", "denominator", "passed", "tier"]
    )
