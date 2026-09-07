"""Case-level (multi-check dossier) delivery date via Gaussian-copula max aggregation.

The naive approach -- and what a UI/backend gap like this usually
defaults to -- is "case P90 = max(check P90 across all checks)". That
is wrong in both directions at once:

* It ignores dependence: checks on the same case often share an
  authority-country risk environment (a national portal outage delays
  every check routed through that country simultaneously), so their
  delays are positively correlated, not independent. Taking the max of
  *marginal* P90s already implicitly (and inconsistently) assumes some
  dependence structure -- it just never states which one.
* It silently drops the extreme-value correction: the maximum of several
  correlated, right-skewed (log-normal-like) durations is itself heavier
  in the right tail than any single component's marginal, which is
  exactly the domain extreme value theory / copulas exist to quantify.

Here each check's fitted P50/P90 quantiles are used to back out
log-normal marginal parameters (mu, sigma), then a Gaussian copula with
a configurable pairwise correlation (same-authority checks correlate
more than cross-authority ones) links the marginals, and the case-level
P50/P80/P90 are read off the empirical distribution of the simulated
per-case maximum. This directly implements the EVT/copula critical-path
propagation requested, without pulling in a heavy dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _lognormal_params_from_quantiles(
    p50: np.ndarray, p90: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p50 = np.maximum(p50, 0.5)
    p90 = np.maximum(p90, p50 + 0.1)
    mu = np.log(p50)
    z90 = norm.ppf(0.90)
    sigma = (np.log(p90) - mu) / z90
    sigma = np.maximum(sigma, 1e-3)
    return mu, sigma


def simulate_case_level_quantiles(
    check_p50: np.ndarray,
    check_p90: np.ndarray,
    same_authority: np.ndarray,
    n_sims: int = 4000,
    rho_same_authority: float = 0.55,
    rho_cross_authority: float = 0.15,
    seed: int = 42,
) -> dict[str, float]:
    """One case's multi-check aggregation.

    `check_p50`/`check_p90`: shape (k,) marginal quantiles per check.
    `same_authority`: shape (k,) group label (e.g. authority_id) used to
    build the pairwise correlation matrix -- checks sharing an authority
    get `rho_same_authority`, others `rho_cross_authority`.
    """
    rng = np.random.default_rng(seed)
    k = len(check_p50)
    if k == 1:
        mu, sigma = _lognormal_params_from_quantiles(check_p50, check_p90)
        draws = rng.lognormal(mu[0], sigma[0], size=n_sims)
    else:
        corr = np.full((k, k), rho_cross_authority)
        for i in range(k):
            for j in range(k):
                if same_authority[i] == same_authority[j]:
                    corr[i, j] = rho_same_authority
        np.fill_diagonal(corr, 1.0)

        z = rng.multivariate_normal(mean=np.zeros(k), cov=corr, size=n_sims)
        u = norm.cdf(z)  # Gaussian copula -> uniform marginals with target correlation

        mu, sigma = _lognormal_params_from_quantiles(check_p50, check_p90)
        marginals = np.exp(mu[None, :] + sigma[None, :] * norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)))
        draws = marginals.max(axis=1)

    return {
        "case_p50": float(np.quantile(draws, 0.50)),
        "case_p80": float(np.quantile(draws, 0.80)),
        "case_p90": float(np.quantile(draws, 0.90)),
        "naive_max_p90": float(np.max(check_p90)),
    }


def aggregate_cases_to_critical_path(
    checks_df,
    case_id_col: str = "case_id",
    authority_col: str = "authority_id",
    p50_col: str = "predicted_p50",
    p90_col: str = "predicted_p90",
    **sim_kwargs,
):
    """Vectorized-over-cases wrapper: groupby case_id, simulate, collect results."""
    import pandas as pd

    rows = []
    for case_id, g in checks_df.groupby(case_id_col):
        result = simulate_case_level_quantiles(
            g[p50_col].to_numpy(),
            g[p90_col].to_numpy(),
            g[authority_col].to_numpy(),
            **sim_kwargs,
        )
        result[case_id_col] = case_id
        result["critical_path_check_idx"] = int(np.argmax(g[p90_col].to_numpy()))
        critical_row = g.iloc[result["critical_path_check_idx"]]
        result["critical_path_check_id"] = critical_row.get("check_id", None)
        rows.append(result)
    return pd.DataFrame(rows)
