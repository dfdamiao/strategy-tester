"""Universe narrowing methods for S5 portfolio construction.

Applied after per-pair backtests, before weighting schemes.
Each function receives the full returns matrix + metadata,
returns a filtered list of pair keys.

References:
    LdP MLAM (2020) Ch.4 — ONC clustering (Ward + silhouette)
    Carver (2015) Ch.8 — Sector grouping, diversification
    Choueifaty & Coignard (2008) — Maximum Diversification Ratio
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import squareform


def _silhouette_score_fast(
    dist_matrix: np.ndarray, labels: np.ndarray,
) -> float:
    """Simplified silhouette score on precomputed distance matrix.

    Avoids sklearn dependency. Computes mean silhouette across all samples.
    """
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    silhouettes = np.zeros(n)
    for i in range(n):
        own_cluster = labels[i]
        own_dists = []
        other_dists: dict[int, list[float]] = {}

        for j in range(n):
            if i == j:
                continue
            if labels[j] == own_cluster:
                own_dists.append(dist_matrix[i, j])
            else:
                other_dists.setdefault(int(labels[j]), []).append(
                    dist_matrix[i, j]
                )

        a_i = np.mean(own_dists) if own_dists else 0.0
        b_i = (
            min(np.mean(ds) for ds in other_dists.values())
            if other_dists
            else 0.0
        )
        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(silhouettes))


def narrow_onc(
    returns_df: pd.DataFrame,
    sharpe_map: dict[str, float],
    max_clusters: int = 80,
    top_per_cluster: int = 3,
    **_kwargs,
) -> list[str]:
    """ONC clustering: group by return correlation, pick best per cluster.

    LdP MLAM (2020) Ch.4: Ward linkage on correlation distance,
    silhouette score for optimal K, top-N per cluster by OOS Sharpe.

    Parameters
    ----------
    returns_df : DataFrame
        Daily returns, columns = pair keys.
    sharpe_map : dict
        pair_key -> OOS Sharpe (for ranking within clusters).
    max_clusters : int
        Upper bound for silhouette scan (default 80).
    top_per_cluster : int
        How many pairs to keep per cluster (default 1).

    Returns
    -------
    list[str] — selected pair keys.
    """
    if len(returns_df.columns) < 4:
        return list(returns_df.columns)

    # Correlation distance matrix (LdP)
    corr = returns_df.corr()
    corr = corr.clip(-1, 1).fillna(0)
    dist = ((1 - corr) / 2) ** 0.5
    np.fill_diagonal(dist.values, 0)

    # Ward linkage
    condensed = squareform(dist.values, checks=False)
    linkage = ward(condensed)

    # Silhouette scan for optimal K
    n = len(returns_df.columns)
    max_k = min(max_clusters, n // 2, n - 1)
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels = fcluster(linkage, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = _silhouette_score_fast(dist.values, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Assign clusters
    labels = fcluster(linkage, t=best_k, criterion="maxclust")
    cluster_map: dict[int, list[str]] = {}
    for pair, label in zip(returns_df.columns, labels):
        cluster_map.setdefault(int(label), []).append(pair)

    # Adaptive top-N: scale per-cluster picks to reach ~target_total
    # LdP MLAM Ch.4 respects silhouette K; we scale picks per cluster
    # to maintain portfolio breadth (Grinold & Kahn: IR = IC × √BR)
    target_total = max(30, n // 5)  # aim for ~20% of universe or 30
    n_clusters = len(cluster_map)
    effective_top = max(
        top_per_cluster,
        (target_total + n_clusters - 1) // n_clusters,  # ceil division
    )

    selected: list[str] = []
    for _label, pairs in sorted(cluster_map.items()):
        ranked = sorted(
            pairs, key=lambda p: sharpe_map.get(p, 0), reverse=True,
        )
        # Don't take more than the cluster has
        selected.extend(ranked[:min(effective_top, len(pairs))])

    return selected


def narrow_sector(
    returns_df: pd.DataFrame,
    sharpe_map: dict[str, float],
    category_map: dict[str, str] | None = None,
    max_sector_pct: float = 0.25,
    **_kwargs,
) -> list[str]:
    """Sector grouping: cap each sector at max_sector_pct of total.

    Carver (2015) Ch.8: no single sector should dominate — cap at 25%.
    Within each sector, keep best by OOS Sharpe up to the cap.
    All sectors represented proportionally.

    Parameters
    ----------
    returns_df : DataFrame
        Daily returns, columns = pair keys.
    sharpe_map : dict
        pair_key -> OOS Sharpe.
    category_map : dict or None
        pair_key -> category string. Unmapped pairs get "Other".
    max_sector_pct : float
        Max fraction of total from one sector (default 0.25, Carver Ch.8).

    Returns
    -------
    list[str] — selected pair keys.
    """
    n_total = len(returns_df.columns)
    if n_total < 4:
        return list(returns_df.columns)

    if category_map is None:
        category_map = {}

    # Group by sector
    sectors: dict[str, list[str]] = {}
    for pair in returns_df.columns:
        cat = category_map.get(pair, "Other")
        sectors.setdefault(cat, []).append(pair)

    # Cap per sector: max 25% of total universe
    max_from_sector = max(2, int(n_total * max_sector_pct))

    selected: list[str] = []
    for _cat, pairs in sorted(sectors.items()):
        ranked = sorted(
            pairs, key=lambda p: sharpe_map.get(p, 0), reverse=True,
        )
        selected.extend(ranked[:max_from_sector])

    return selected


def narrow_max_div(
    returns_df: pd.DataFrame,
    sharpe_map: dict[str, float],
    max_n: int = 100,
    min_dr_improvement: float = 0.005,
    **_kwargs,
) -> list[str]:
    """Max diversification ratio: greedy forward selection.

    Choueifaty & Coignard (2008): iteratively add the asset that
    maximises the portfolio diversification ratio:
        DR = (w' * sigma) / sqrt(w' * Sigma * w)
    where sigma = vector of individual vols, Sigma = covariance matrix.

    Stops when DR improvement < min_dr_improvement (diminishing returns)
    or max_n reached. No arbitrary target — the DR curve decides.

    Parameters
    ----------
    returns_df : DataFrame
        Daily returns, columns = pair keys.
    sharpe_map : dict
        pair_key -> OOS Sharpe (used to seed + break ties).
    max_n : int
        Hard upper bound (default 100, safety cap).
    min_dr_improvement : float
        Stop when marginal DR gain < this (default 0.005 = 0.5%).

    Returns
    -------
    list[str] — selected pair keys.
    """
    n = len(returns_df.columns)
    if n <= 3:
        return list(returns_df.columns)

    # Compute covariance and individual vols
    cov = returns_df.cov().values
    vols = np.sqrt(np.diag(cov))
    pairs = list(returns_df.columns)

    # Start with highest Sharpe pair
    selected_idx: list[int] = [
        max(range(n), key=lambda i: sharpe_map.get(pairs[i], 0)),
    ]
    remaining = set(range(n)) - set(selected_idx)
    prev_dr = 1.0  # DR of a single asset = 1.0

    # Greedy forward selection — stop when DR plateaus
    while len(selected_idx) < min(max_n, n) and remaining:
        best_dr = -np.inf
        best_j = -1
        for j in remaining:
            trial = selected_idx + [j]
            k = len(trial)
            w = np.ones(k) / k
            sub_vols = vols[trial]
            sub_cov = cov[np.ix_(trial, trial)]
            port_vol = np.sqrt(w @ sub_cov @ w)
            if port_vol < 1e-12:
                continue
            dr = (w @ sub_vols) / port_vol
            if dr > best_dr:
                best_dr = dr
                best_j = j
        if best_j < 0:
            break
        # Stop if DR improvement is negligible
        dr_improvement = (best_dr - prev_dr) / prev_dr if prev_dr > 0 else 0
        if len(selected_idx) >= 5 and dr_improvement < min_dr_improvement:
            break
        selected_idx.append(best_j)
        prev_dr = best_dr
        remaining.discard(best_j)

    return [pairs[i] for i in selected_idx]


# Registry of all narrowing methods
NARROWING_METHODS: dict[str, object] = {
    "all": None,  # No narrowing — use all pairs
    "onc": narrow_onc,
    "sector": narrow_sector,
    "max_div": narrow_max_div,
}
