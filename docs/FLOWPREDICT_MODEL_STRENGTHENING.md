# FlowPredict TAT Model — Strengthening Notes

Companion to `src/flowpredict/`. Covers the gap analysis between the
`Master_geminicode` prototype / FlowPredict frontend schema and the
strengthened backend pipeline now in this repo.

## 1. Bottlenecks & risks found in the prototype

- **Data leakage**: `rework_impact_score = qc_bounce_count * 8.4` reused
  the exact coefficient used to synthesize the label — the model was
  partially memorizing its own generating function. Fixed: `qc_bounce_count`
  kept raw in `features.py`, no hand-computed multiplier feature.
- **No hierarchy**: one flat table, no authority/country/check-type
  entities, so nothing could be linked to disruption signals or closures
  by (country, authority, date window). Fixed in `synthetic_data.py`.
- **No censoring**: every training/eval row was a *closed* case. In
  production a large share of the portfolio is open (right-censored),
  and excluding it biases the model toward faster-resolving cases.
  Fixed via IPCW in `survival.py`.
- **Real bug**: the original script's calibration-set isotonic clamp
  call passed `raw_p80_calib` in place of a genuine P90 calibration
  prediction (`models[0.90]` was never scored on `X_calib`). Harmless in
  that script only because the discarded value was never used downstream
  — but it meant no P90 conformal calibration was actually possible from
  that code as written.
- **Single global conformal q_hat**: hides subgroup miscoverage (e.g. one
  authority category systematically under-covered while another
  over-covers, netting out to a passing global number). Fixed with
  Mondrian (grouped) conformal in `quantile_model.py`.
- **Case-level dates via naive `max()` of check-level quantiles**: ignores
  dependence between checks on the same case and the extreme-value effect
  of taking a max. Fixed with copula-based aggregation in `critical_path.py`.

## 2. Linking disruption signals to hierarchical entities

`synthetic_data.py` generates an independent disruption-signal feed keyed
on `authority_id`/`country`, each with a `start_day`, `severity`, and a
severity-specific half-life. `features.py` joins signals onto a case by
authority + time-window, decays severity exponentially both from signal
onset and from the case's own elapsed silence, and forms an interaction
term (`closure_x_disruption`) with the gazetted-closure calendar. This
generalizes directly to the frontend's `DisruptionSignal` type — join key
should be `(authorityId | country, dateReported)`, not a manually
maintained per-case flag.

## 3. Survival analysis + quantile regression + conformal

`survival.py` implements Kaplan-Meier on the censoring indicator and
derives IPCW sample weights, following the Portnoy censored-quantile-
regression bridge: closed cases are up-weighted by `1/G(t-)` so they
statistically stand in for the open cases that can't contribute a direct
label. `quantile_model.py` trains LightGBM `objective='quantile'` models
with those weights, isotonic-clamps for monotonicity, then applies
Mondrian split-conformal calibration per authority-category tier for a
coverage guarantee that holds per subgroup, not just on average.

## 4. Feature pruning & interaction terms

`features.prune_collinear_features` runs iterative VIF and drops
`silence_ratio` (VIF ≈ 43 against `authority_encoded_median_tat` and
`days_without_response`, which already jointly encode the same signal) —
this replaces eyeballing a correlation matrix. Interaction terms added:
`silence_ratio` (before pruning) / `bayesian_stalled_prob`,
`disruption_severity_decay`, and `closure_x_disruption`.

## 5. Critical-path propagation

`critical_path.simulate_case_level_quantiles` fits log-normal marginals
per check from its P50/P90, links them with a Gaussian copula
(same-authority ρ=0.55, cross-authority ρ=0.15), and reads case-level
P50/P80/P90 off the simulated max. The pipeline prints the mean gap
between this and the naive `max(check P90)` heuristic so the correction
is directly observable.

## 6. Frontend/backend schema recommendations

`src/data/types.ts` (frontend) already anticipates most of this —
`BayesianSurvivalTransition`, `PredictiveConfidenceScore`,
`ModelMetadata.validationMetrics` — which is good; the gap is that the
current backend never produced numbers matching that schema's intent.
Concrete changes:

- **Add**, on `VerificationCase`: a `censoringStatus: 'closed' | 'open'`
  field and `ipcwWeight?: number` so the UI can show *why* an open case's
  bounds differ from a superficially similar closed one.
- **Add**, on `ModelMetadata.validationMetrics`: per-authority-category
  coverage (`minSubgroupCoveragePct`/`maxSubgroupCoveragePct`) — the
  pipeline already computes this; surfacing it stops a well-calibrated
  global number from hiding a miscalibrated tier.
- **Remove/rename**: nothing in `types.ts` needs removing — it's ahead
  of the old backend, not behind it.
- **Refactor** `VerificationCase.isCriticalPath` / `criticalPathCheckId`:
  populate from `critical_path.aggregate_cases_to_critical_path`'s
  `critical_path_check_id` output instead of a heuristic, and add
  `casePredictedP90NaiveMaxDelta` so a reviewer can see the copula
  correction size per case.
- **Python inference pipeline**: replace any prior "max of check P90"
  case-date logic with `critical_path.aggregate_cases_to_critical_path`,
  and gate any TAT training job on `event_observed`/`observed_days`
  columns being present (fail fast rather than silently training
  closed-only).
