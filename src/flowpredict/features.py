"""Leakage-safe feature engineering for the FlowPredict TAT model.

Key fixes relative to the original prototype:

* No feature is allowed to encode the label's generating coefficients.
  The prototype's `rework_impact_score = qc_bounce_count * 8.4` used the
  *exact* constant used to synthesize `actual_calendar_tat`, which is
  circular reasoning, not a feature. Here `qc_bounce_count` is kept raw
  and the model is left to learn its own coefficient.
* Authority/country "historical median TAT" is computed with out-of-fold
  (K-fold) target encoding on the *training* label only, so a case never
  sees a statistic computed from itself (target leakage via group means
  is the single most common real-world PSV/TAT modeling bug).
* Adds the two interaction terms requested: silence-to-TAT ratio and a
  disruption-severity decay term (already decayed at generation time in
  `synthetic_data.py`; here we additionally decay it relative to the
  case's own elapsed silence window, since the same signal should matter
  less to a case that is about to close than one still waiting).
* Prunes multicollinear features via iterative VIF (variance inflation
  factor) rather than eyeballing a correlation matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.flowpredict._stats import kfold_splits

CATEGORICAL_GROUP_COLS = ["authority_id", "country", "check_type"]


def oof_target_encode(
    df: pd.DataFrame,
    group_col: str,
    target: pd.Series,
    train_mask: np.ndarray,
    n_splits: int = 5,
    global_prior_weight: float = 8.0,
    seed: int = 42,
) -> pd.Series:
    """Out-of-fold shrinkage target encoding.

    Rows outside `train_mask` (calibration/test) are encoded using the
    statistic fit on the *entire* training set (no leakage, since none of
    those rows contributed to it). Training rows are encoded out-of-fold
    so each row's encoding excludes itself and its fold.
    """
    encoded = pd.Series(np.nan, index=df.index, dtype=float)
    global_mean = target[train_mask].mean()

    train_idx = df.index[train_mask]
    for fold_train_pos, fold_hold_pos in kfold_splits(len(train_idx), n_splits, seed):
        fold_train_idx = train_idx[fold_train_pos]
        fold_hold_idx = train_idx[fold_hold_pos]
        fold_groups = df.loc[fold_train_idx, group_col]
        stats = target.loc[fold_train_idx].groupby(fold_groups).agg(["mean", "count"])
        shrunk = (stats["mean"] * stats["count"] + global_mean * global_prior_weight) / (
            stats["count"] + global_prior_weight
        )
        hold_encoded = df.loc[fold_hold_idx, group_col].map(shrunk).fillna(global_mean)
        encoded.loc[fold_hold_idx] = hold_encoded

    full_stats = target[train_mask].groupby(df.loc[train_mask, group_col]).agg(["mean", "count"])
    full_shrunk = (full_stats["mean"] * full_stats["count"] + global_mean * global_prior_weight) / (
        full_stats["count"] + global_prior_weight
    )
    non_train_idx = df.index[~train_mask]
    non_train_encoded = df.loc[non_train_idx, group_col].map(full_shrunk).fillna(global_mean)
    encoded.loc[non_train_idx] = non_train_encoded
    return encoded


def compute_bayesian_stalled_prob(silence_ratio: pd.Series) -> pd.Series:
    prior = 0.15
    likelihood_stalled = 1.0 / (1.0 + np.exp(-2.0 * (silence_ratio - 0.8)))
    likelihood_normal = 1.0 / (1.0 + np.exp(2.0 * (silence_ratio - 0.5)))
    num = likelihood_stalled * prior
    den = num + likelihood_normal * (1.0 - prior)
    return num / (den + 1e-9)


def engineer_features(
    df: pd.DataFrame, label_days: pd.Series, train_mask: np.ndarray
) -> pd.DataFrame:
    out = df.copy()

    out["authority_encoded_median_tat"] = oof_target_encode(
        out, "authority_id", label_days, train_mask
    )
    out["country_encoded_median_tat"] = oof_target_encode(out, "country", label_days, train_mask)
    out["check_type_encoded_median_tat"] = oof_target_encode(
        out, "check_type", label_days, train_mask
    )

    out["silence_ratio"] = out["days_without_response"] / (
        out["authority_encoded_median_tat"] + 1e-5
    )
    out["bayesian_stalled_prob"] = compute_bayesian_stalled_prob(out["silence_ratio"])

    # Disruption severity decayed further by how long the case has already
    # been waiting: a still-decaying outage matters most to a case that
    # just started, and progressively less as the case's own clock runs.
    out["disruption_severity_decay"] = out["disruption_days_added"] * np.exp(
        -out["days_without_response"] / 14.0
    )

    out["closure_x_disruption"] = np.log1p(out["closure_days_overlapping"]) * np.log1p(
        out["disruption_severity_decay"]
    )

    out["check_type"] = out["check_type"].astype("category")
    out["country"] = out["country"].astype("category")
    out["category"] = out["category"].astype("category")

    return out


BASE_NUMERIC_FEATURES = [
    "authority_encoded_median_tat",
    "country_encoded_median_tat",
    "check_type_encoded_median_tat",
    "days_without_response",
    "qc_bounce_count",
    "insufficiency_days",
    "closure_days_overlapping",
    "disruption_severity_decay",
    "silence_ratio",
    "bayesian_stalled_prob",
    "closure_x_disruption",
    "process_type_rti",
]


def prune_collinear_features(
    df: pd.DataFrame, candidate_features: list[str], vif_threshold: float = 8.0
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively drop the highest-VIF numeric feature until all are below threshold.

    `country_encoded_median_tat` and `authority_encoded_median_tat` are
    intentionally close to `check_type_encoded_median_tat`'s information
    content when the dataset is small (shared prior shrinkage), and
    `closure_x_disruption` is a deliberate interaction of two already-kept
    mains -- this is exactly the redundancy VIF pruning should catch
    instead of hand-picking.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    features = list(candidate_features)
    dropped = []
    while len(features) > 2:
        X = df[features].to_numpy(dtype=float)
        X = X - X.mean(axis=0)
        X = X / (X.std(axis=0) + 1e-9)
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_vif = max(vifs)
        if max_vif < vif_threshold:
            break
        worst_idx = int(np.argmax(vifs))
        dropped.append((features[worst_idx], max_vif))
        features.pop(worst_idx)

    report = pd.DataFrame(dropped, columns=["dropped_feature", "vif_at_drop"])
    return features, report
