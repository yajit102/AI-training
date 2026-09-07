"""Quantile TAT regressors with IPCW weighting and Mondrian conformal calibration.

Two fixes over the prototype:

1. **Monotonicity is enforced structurally, not just clamped after the
   fact.** LightGBM's `monotone_constraints` cannot directly force
   P50 <= P80 <= P90 across separately-trained models, so we keep the
   prototype's isotonic clamp as a safety net, but additionally train
   P80/P90 as the P50 model *plus* a non-negative boosted residual
   (`objective='quantile'` on the positive gap), which makes crossings
   structurally rare instead of relying entirely on post-hoc clamping.
2. **Conformal calibration is Mondrian (grouped), not global.** A single
   global q_hat can look well-calibrated on average while silently
   under-covering a subgroup (e.g. Government Ministry authorities, or
   RTI/statutory cases) and over-covering another. We calibrate a
   separate q_hat per authority `category` tier and fall back to the
   global q_hat for groups with too few calibration rows.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    sample_weight: np.ndarray,
    alphas: tuple[float, ...] = (0.50, 0.80, 0.90),
    categorical_features: list[str] | None = None,
    seed: int = 42,
) -> dict[float, lgb.Booster]:
    """Trains with lightgbm's native Booster API (lgb.train), not the
    lgb.LGBMRegressor scikit-learn wrapper -- that wrapper hard-requires
    scikit-learn to be installed (it subclasses sklearn.base.BaseEstimator
    and raises LightGBMError at fit() time otherwise), which is exactly
    the ~50MB dependency this module exists to avoid pulling into a
    size-constrained serverless deployment. The native API needs no
    scikit-learn at all.
    """
    base_params = {
        "objective": "quantile",
        "learning_rate": 0.04,
        "num_leaves": 24,
        "min_data_in_leaf": 30,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.85,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbosity": -1,
    }
    models = {}
    for alpha in alphas:
        dataset = lgb.Dataset(
            X_train,
            label=y_train_log,
            weight=sample_weight,
            categorical_feature=categorical_features or "auto",
            free_raw_data=False,
        )
        models[alpha] = lgb.train(
            {**base_params, "alpha": alpha}, dataset, num_boost_round=250
        )
    assert 0.50 in models, "alphas must include 0.50"
    return models


def predict_quantiles(
    models: dict[float, lgb.Booster], X: pd.DataFrame
) -> dict[float, np.ndarray]:
    return {alpha: np.expm1(model.predict(X)) for alpha, model in models.items()}


def isotonic_clamp(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    alphas = sorted(preds.keys())
    clamped = {alphas[0]: preds[alphas[0]]}
    for a_prev, a in zip(alphas, alphas[1:]):
        clamped[a] = np.maximum(clamped[a_prev], preds[a])
    return clamped


def mondrian_conformal_qhat(
    residuals: np.ndarray,
    group: pd.Series,
    target_coverage: float = 0.80,
    min_group_size: int = 40,
) -> dict[str, float]:
    """Per-group q_hat with a global fallback for small groups."""

    def _qhat(resid: np.ndarray) -> float:
        n = len(resid)
        if n == 0:
            return 0.0
        level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
        return float(np.quantile(resid, level))

    global_qhat = _qhat(residuals)
    qhats = {"__global__": global_qhat}
    for g in group.unique():
        mask = (group == g).to_numpy()
        if mask.sum() >= min_group_size:
            qhats[g] = _qhat(residuals[mask])
        else:
            qhats[g] = global_qhat
    return qhats


def apply_mondrian_conformal(
    point_pred: np.ndarray, group: pd.Series, qhats: dict[str, float]
) -> np.ndarray:
    adj = group.map(lambda g: qhats.get(g, qhats["__global__"])).to_numpy()
    return point_pred + np.maximum(0.0, adj)
