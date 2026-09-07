"""Synthetic PSV (Primary Source Verification) check dataset.

Unlike the original prototype (a single flat table with no hierarchy),
this generator builds the entity hierarchy the FlowPredict frontend
schema actually models: Country -> Issuing Authority -> Check, plus an
independent disruption-signal feed and a gazetted-closure calendar that
are joined onto checks by (country, authority, date window) rather than
baked directly into the label. Roughly `open_fraction` of checks are
right-censored ("Open" in the UI): we only observe elapsed days so far,
not the final TAT, which is the actual real-world data-generating
process and something the original script ignored entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["India", "Philippines", "Egypt", "Nigeria", "Pakistan", "Jordan"]
CHECK_TYPES = ["Education", "Health License", "Employment", "Good Standing", "Identity"]
CATEGORIES = ["University", "Medical Council", "Government Ministry", "Hospital/Employer"]


def _build_authorities(n_authorities: int, rng: np.random.Generator) -> pd.DataFrame:
    country = rng.choice(COUNTRIES, size=n_authorities)
    category = rng.choice(CATEGORIES, size=n_authorities)
    # Authority-level baseline latency: the true (unobserved in production)
    # median TAT an authority tends toward. Bureaucratic ministries and
    # medical councils run slower than university registrars.
    category_shift = pd.Series(category).map(
        {
            "University": 0.0,
            "Hospital/Employer": -1.5,
            "Medical Council": 4.0,
            "Government Ministry": 6.0,
        }
    ).to_numpy()
    base_median = np.clip(
        rng.gamma(shape=4.0, scale=3.0, size=n_authorities) + category_shift, 3.0, 45.0
    )
    volatility = rng.uniform(0.15, 0.55, size=n_authorities)  # authority-specific spread
    disruption_prone = rng.uniform(0.0, 1.0, size=n_authorities) < 0.3
    return pd.DataFrame(
        {
            "authority_id": [f"IA-{i:04d}" for i in range(n_authorities)],
            "country": country,
            "category": category,
            "authority_true_median_tat": base_median,
            "authority_volatility": volatility,
            "authority_disruption_prone": disruption_prone,
        }
    )


def _build_disruption_signals(
    authorities: pd.DataFrame, rng: np.random.Generator, horizon_days: int
) -> pd.DataFrame:
    """Independent event feed: portal outages, unrest, strikes.

    Only authorities flagged disruption-prone get signals, and signals are
    sparse in time -- this is what lets the decay feature (point 4) carry
    real information instead of being a restatement of the label.
    """
    prone = authorities[authorities["authority_disruption_prone"]]
    rows = []
    for _, a in prone.iterrows():
        n_signals = rng.poisson(1.2)
        for _ in range(n_signals):
            start = rng.integers(0, horizon_days)
            severity = rng.choice(["Warning", "High", "Severe"], p=[0.5, 0.35, 0.15])
            half_life = {"Warning": 2.0, "High": 5.0, "Severe": 9.0}[severity]
            rows.append(
                {
                    "authority_id": a["authority_id"],
                    "country": a["country"],
                    "category": rng.choice(
                        [
                            "Ministry Portal Outage",
                            "Academic Strike",
                            "Courier/Postal Disruption",
                            "Weather/Disaster",
                        ]
                    ),
                    "severity": severity,
                    "severity_weight": {"Warning": 1.0, "High": 2.2, "Severe": 3.8}[severity],
                    "start_day": start,
                    "decay_half_life_days": half_life,
                }
            )
    return pd.DataFrame(rows)


def _build_closures(
    authorities: pd.DataFrame, rng: np.random.Generator, horizon_days: int
) -> pd.DataFrame:
    rows = []
    for country in authorities["country"].unique():
        n_closures = rng.integers(2, 5)
        for _ in range(n_closures):
            start = rng.integers(0, horizon_days)
            span = rng.choice([1, 2, 4, 7, 10])
            rows.append({"country": country, "start_day": start, "closure_days": span})
    return pd.DataFrame(rows)


def generate_hierarchical_psv_dataset(
    n_cases: int = 6000,
    n_authorities: int = 60,
    open_fraction: float = 0.22,
    horizon_days: int = 240,
    seed: int = 42,
):
    """Returns (cases_df, authorities_df, disruptions_df, closures_df).

    cases_df carries both the ground-truth `true_calendar_tat` (for
    evaluation/debugging only) and the production-realistic observed
    columns: `observed_days`, `event_observed` (1 = closed/verified,
    0 = still open / right-censored at the snapshot date).
    """
    rng = np.random.default_rng(seed)
    authorities = _build_authorities(n_authorities, rng)
    disruptions = _build_disruption_signals(authorities, rng, horizon_days)
    closures = _build_closures(authorities, rng, horizon_days)

    auth_idx = rng.integers(0, n_authorities, size=n_cases)
    case_authority = authorities.iloc[auth_idx].reset_index(drop=True)

    check_type = rng.choice(CHECK_TYPES, size=n_cases)
    process_type_rti = rng.choice([0, 1], p=[0.85, 0.15], size=n_cases)
    start_day = rng.integers(0, horizon_days - 20, size=n_cases)

    # Silence / QC dynamics -- operational, not authority-intrinsic.
    days_without_response = rng.exponential(scale=5.5, size=n_cases)
    qc_bounce_count = rng.choice([0, 1, 2], p=[0.72, 0.20, 0.08], size=n_cases)
    insufficiency_days = rng.choice([0, 0, 0, 3, 6], size=n_cases).astype(float)

    check_type_shift = pd.Series(check_type).map(
        {
            "Education": 0.0,
            "Health License": 3.0,
            "Employment": -2.0,
            "Good Standing": 1.0,
            "Identity": -3.0,
        }
    ).to_numpy()

    # Disruption exposure at case start: decayed severity of any signal
    # active on/near the case's authority within its lifetime window.
    disruption_days_added = np.zeros(n_cases)
    for i in range(n_cases):
        a_id = case_authority.loc[i, "authority_id"]
        sigs = disruptions[disruptions["authority_id"] == a_id]
        if sigs.empty:
            continue
        elapsed = start_day[i] - sigs["start_day"].to_numpy()
        active = elapsed >= 0
        decay = sigs["severity_weight"].to_numpy()[active] * np.exp(
            -elapsed[active] / sigs["decay_half_life_days"].to_numpy()[active]
        )
        disruption_days_added[i] = decay.sum() * 1.8

    closure_days_overlapping = np.zeros(n_cases)
    for i in range(n_cases):
        country = case_authority.loc[i, "country"]
        cls = closures[closures["country"] == country]
        overlap = cls[(cls["start_day"] >= start_day[i]) & (cls["start_day"] <= start_day[i] + 40)]
        closure_days_overlapping[i] = overlap["closure_days"].sum()

    rti_penalty = process_type_rti * rng.uniform(4.0, 10.0, size=n_cases)

    true_calendar_tat = (
        case_authority["authority_true_median_tat"].to_numpy()
        + check_type_shift
        + days_without_response * 0.8
        # noisy per-bounce penalty, NOT a fixed constant the model could memorize
        + qc_bounce_count * rng.normal(7.5, 1.5, size=n_cases).clip(min=2.0)
        + insufficiency_days
        + closure_days_overlapping
        + disruption_days_added
        + rti_penalty
        + case_authority["authority_true_median_tat"].to_numpy()
        * case_authority["authority_volatility"].to_numpy()
        * rng.standard_normal(n_cases).clip(-1.5, 3.0)
    )
    true_calendar_tat = np.clip(true_calendar_tat, 2.0, None)

    snapshot_day = horizon_days
    completion_day = start_day + true_calendar_tat
    is_open = rng.uniform(0.0, 1.0, size=n_cases) < open_fraction
    is_open = is_open | (completion_day > snapshot_day)
    observed_days = np.where(is_open, np.maximum(snapshot_day - start_day, 0.1), true_calendar_tat)
    event_observed = np.where(is_open, 0, 1)

    cases = pd.DataFrame(
        {
            "case_id": [f"CASE-{i:06d}" for i in range(n_cases)],
            "authority_id": case_authority["authority_id"].to_numpy(),
            "country": case_authority["country"].to_numpy(),
            "category": case_authority["category"].to_numpy(),
            "check_type": check_type,
            "process_type_rti": process_type_rti,
            "days_without_response": days_without_response,
            "qc_bounce_count": qc_bounce_count,
            "insufficiency_days": insufficiency_days,
            "closure_days_overlapping": closure_days_overlapping,
            "disruption_days_added": disruption_days_added,
            "true_calendar_tat": true_calendar_tat,
            "observed_days": observed_days,
            "event_observed": event_observed,
        }
    )
    return cases, authorities, disruptions, closures
