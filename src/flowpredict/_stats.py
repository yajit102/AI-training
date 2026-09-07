"""Pure-numpy standard-normal CDF/PPF, train/test split, K-fold, and
quantile-regression metrics.

Exists to avoid hard `scipy`/`scikit-learn` dependencies: together they
add ~163MB unzipped and, combined with numpy+pandas+lightgbm, pushed the
Vercel serverless function's bundle to ~290MB -- over the platform's
250MB unzipped function-size limit, which is what caused the deploy to
fail. Everything actually used from those two libraries (normal
CDF/PPF, a random train/test split, K-fold index generation, MAE, and
pinball loss) is a few lines of numpy, so it's reimplemented directly
rather than pulling in ~163MB for it.
"""
from __future__ import annotations

import numpy as np


def train_test_split_indices(
    indices: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to sklearn.model_selection.train_test_split on an index array."""
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    perm = rng.permutation(len(indices))
    n_test = int(round(len(indices) * test_size))
    test_idx = indices[perm[:n_test]]
    train_idx = indices[perm[n_test:]]
    return train_idx, test_idx


def kfold_splits(n: int, n_splits: int, seed: int):
    """Equivalent to sklearn.model_selection.KFold(shuffle=True).split(range(n))."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        val_pos = order[current : current + fold_size]
        train_pos = np.concatenate([order[:current], order[current + fold_size :]])
        yield train_pos, val_pos
        current += fold_size


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mean_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Equivalent to sklearn.metrics.mean_pinball_loss."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via the erf identity (numpy has no erf, but
    numpy's polynomial/vectorized ops make a direct Abramowitz & Stegun
    7.1.26 approximation to erf easy; ~1.5e-7 max absolute error)."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x) / np.sqrt(2.0)

    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf_approx = 1.0 - poly * np.exp(-ax * ax)

    return 0.5 * (1.0 + sign * erf_approx)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Standard normal inverse CDF (quantile function) via Acklam's
    rational approximation; relative error ~1.15e-9, standard and widely
    used where a scipy dependency isn't wanted."""
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    p_low = 0.02425
    p_high = 1 - p_low

    result = np.empty_like(p)

    low_mask = p < p_low
    if np.any(low_mask):
        q = np.sqrt(-2 * np.log(p[low_mask]))
        result[low_mask] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    mid_mask = (~low_mask) & (p <= p_high)
    if np.any(mid_mask):
        q = p[mid_mask] - 0.5
        r = q * q
        result[mid_mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

    high_mask = p > p_high
    if np.any(high_mask):
        q = np.sqrt(-2 * np.log(1 - p[high_mask]))
        result[high_mask] = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    return result
