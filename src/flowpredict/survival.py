"""Right-censoring-aware weighting for open (still-in-progress) cases.

The original prototype trained and evaluated only on cases that had
already closed -- it never accounted for the fact that, at prediction
time in production, a large share of the portfolio is *open*: the
final TAT is unknown and only a lower bound (days elapsed so far) is
observed. Training only on closed cases biases the model toward
"easy", faster-resolving cases and systematically understates TAT/risk
for the harder, slower-moving open population (a textbook informative
right-censoring problem).

We use Inverse Probability of Censoring Weighting (IPCW), the standard
bridge between survival analysis and pinball-loss quantile regression
(Portnoy 2003 censored quantile regression is the theoretical basis):

1. Fit a Kaplan-Meier estimate of the *censoring* distribution G(t)
   (i.e. treat "still open" as the event of interest for this
   auxiliary model).
2. Weight each closed case's row by 1 / G(observed_days-), so a closed
   case observed at a time when few peers are still open (i.e. it
   "survived" a long censoring-prone window before closing) is
   up-weighted to stand in for the open cases that would otherwise be
   dropped.
3. Open cases are excluded from the direct pinball-loss target (their
   true TAT is unknown) but are used to fit G(t) itself, and their
   elapsed-days lower bound is later used as a hard floor when serving
   predictions (see `pipeline.py`).

No external survival-analysis dependency (lifelines) is required --
Kaplan-Meier is ~15 lines of numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(observed_days: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Standard product-limit estimator.

    Returns a dataframe of (time, n_at_risk, n_events, survival) sorted
    by time, where `survival` is S(t) for the given `event` indicator
    (call twice, swapping `event` <-> `1 - event`, to get both the TAT
    survival curve and the censoring distribution G(t)).
    """
    order = np.argsort(observed_days)
    t = observed_days[order]
    e = event[order]
    unique_times = np.unique(t)

    surv = 1.0
    rows = []
    for ut in unique_times:
        at_risk = np.sum(t >= ut)
        events_here = np.sum((t == ut) & (e == 1))
        if at_risk > 0:
            surv *= 1.0 - events_here / at_risk
        rows.append((ut, at_risk, events_here, surv))
    return pd.DataFrame(rows, columns=["time", "n_at_risk", "n_events", "survival"])


def censoring_distribution(observed_days: np.ndarray, event_observed: np.ndarray) -> pd.DataFrame:
    """G(t): Kaplan-Meier on the *censoring* indicator (event=1-observed)."""
    return kaplan_meier(observed_days, 1 - event_observed)


def ipcw_weights(
    observed_days: np.ndarray, event_observed: np.ndarray, clip_max: float = 20.0
) -> np.ndarray:
    """1 / G(observed_days-) for observed (closed) cases; 0 for censored (open) rows.

    Weights are clipped to avoid a handful of very-late closures (where
    G(t) is estimated from few remaining at-risk cases and is noisy)
    from dominating the loss.
    """
    g_curve = censoring_distribution(observed_days, event_observed)
    # G(t-): survival just *before* each time -- avoids dividing by G(t)
    # itself dropping to (near) zero exactly at large observed events.
    times = g_curve["time"].to_numpy()
    surv = g_curve["survival"].to_numpy()
    g_before = np.concatenate([[1.0], surv[:-1]])
    g_lookup = dict(zip(times, g_before))

    weights = np.zeros_like(observed_days, dtype=float)
    for i, (t, e) in enumerate(zip(observed_days, event_observed)):
        if e == 1:
            g_t = g_lookup.get(t, 1.0)
            weights[i] = 1.0 / max(g_t, 1e-3)
    weights = np.clip(weights, 0.0, clip_max)
    return weights
