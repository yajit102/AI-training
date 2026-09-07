"""End-to-end FlowPredict TAT pipeline: the strengthened successor to
`Master_geminicode`.

Run via `python scripts/train_flowpredict_tat_model.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

from src.flowpredict import critical_path, features, survival
from src.flowpredict.quantile_model import (
    apply_mondrian_conformal,
    fit_quantile_models,
    isotonic_clamp,
    mondrian_conformal_qhat,
    predict_quantiles,
)
from src.flowpredict.synthetic_data import generate_hierarchical_psv_dataset


def run(
    n_cases: int = 8000, seed: int = 42, verbose: bool = True, skip_vif: bool = False
) -> dict:
    cases, authorities, disruptions, closures = generate_hierarchical_psv_dataset(
        n_cases=n_cases, seed=seed
    )

    closed = cases[cases["event_observed"] == 1].reset_index(drop=True)
    open_cases = cases[cases["event_observed"] == 0].reset_index(drop=True)

    # Three-way split on CLOSED cases only (open cases have no usable label
    # for pinball loss and are folded in purely via IPCW weighting below).
    idx = np.arange(len(closed))
    idx_train, idx_temp = train_test_split(idx, test_size=0.4, random_state=seed)
    idx_calib, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=seed)

    train_mask_full = np.zeros(len(closed), dtype=bool)
    train_mask_full[idx_train] = True

    label_days = closed["true_calendar_tat"]
    engineered = features.engineer_features(closed, label_days, train_mask_full)

    candidate_features = features.BASE_NUMERIC_FEATURES
    if skip_vif:
        # VIF pruning (statsmodels) is the slowest single step and not
        # worth paying for on every request in a latency-sensitive serving
        # path (e.g. a serverless endpoint) -- keep the previously-known
        # pruned set instead. Always run it at least once offline (CLI
        # default) and hardcode the result here if it changes.
        pruned_features = [f for f in candidate_features if f != "silence_ratio"]
        vif_report = pd.DataFrame(columns=["dropped_feature", "vif_at_drop"])
    else:
        pruned_features, vif_report = features.prune_collinear_features(
            engineered.loc[train_mask_full], candidate_features
        )

    cat_cols = [c for c in ["check_type", "country", "category"] if c in engineered.columns]
    model_features = pruned_features + cat_cols

    X = engineered[model_features]
    y_log = np.log1p(closed["true_calendar_tat"])

    X_train, y_train = X.iloc[idx_train], y_log.iloc[idx_train]
    X_calib, y_calib = X.iloc[idx_calib], y_log.iloc[idx_calib]
    X_test, y_test = X.iloc[idx_test], y_log.iloc[idx_test]

    # IPCW: weight training rows by inverse censoring-survival probability
    # using the FULL dataset's censoring pattern (open cases inform G(t)
    # even though they never enter X_train directly).
    ipcw_all = survival.ipcw_weights(
        cases["observed_days"].to_numpy(), cases["event_observed"].to_numpy()
    )
    ipcw_train = ipcw_all[cases["event_observed"].to_numpy() == 1][idx_train]
    # closed rows should be > 0 by construction; guard against a stray edge case
    ipcw_train = np.where(ipcw_train <= 0, 1.0, ipcw_train)

    models = fit_quantile_models(
        X_train, y_train, sample_weight=ipcw_train, categorical_features=cat_cols, seed=seed
    )

    calib_preds = isotonic_clamp(predict_quantiles(models, X_calib))
    test_preds = isotonic_clamp(predict_quantiles(models, X_test))

    actuals_calib = np.expm1(y_calib.to_numpy())
    actuals_test = np.expm1(y_test.to_numpy())

    group_calib = engineered.loc[X_calib.index, "category"].astype(str)
    group_test = engineered.loc[X_test.index, "category"].astype(str)

    conformal_test = {}
    qhat_reports = {}
    for alpha in (0.50, 0.80, 0.90):
        resid = actuals_calib - calib_preds[alpha]
        qhats = mondrian_conformal_qhat(resid, group_calib, target_coverage=alpha)
        conformal_test[alpha] = apply_mondrian_conformal(test_preds[alpha], group_test, qhats)
        qhat_reports[alpha] = qhats

    conformal_test = isotonic_clamp(conformal_test)

    diagnostics = _build_diagnostics(
        actuals_test, test_preds, conformal_test, group_test, y_test, X_test
    )

    # --- Critical-path demo: fan each test case out into 1-3 synthetic
    # checks (reusing its own quantile forecast plus jittered siblings)
    # and aggregate via the copula method instead of naive max().
    cp_input = _build_synthetic_multicheck_frame(
        engineered, X_test, test_preds, conformal_test, seed
    )
    case_level = critical_path.aggregate_cases_to_critical_path(cp_input)

    if verbose:
        _print_report(diagnostics, vif_report, model_features, case_level, qhat_reports)

    return {
        "diagnostics": diagnostics,
        "vif_report": vif_report,
        "model_features": model_features,
        "case_level_critical_path": case_level,
        "qhat_reports": qhat_reports,
        "n_open_cases": len(open_cases),
        "n_closed_cases": len(closed),
    }


def _build_diagnostics(
    actuals_test, test_preds, conformal_test, group_test, y_test, X_test
) -> pd.DataFrame:
    rows = []
    for alpha in (0.50, 0.80, 0.90):
        raw_cov = float(np.mean(actuals_test <= test_preds[alpha]) * 100)
        conf_cov = float(np.mean(actuals_test <= conformal_test[alpha]) * 100)
        pinball_raw = mean_pinball_loss(actuals_test, test_preds[alpha], alpha=alpha)
        pinball_conf = mean_pinball_loss(actuals_test, conformal_test[alpha], alpha=alpha)
        # worst-case subgroup coverage gap: what a single global q_hat would hide
        covered = pd.Series(actuals_test <= conformal_test[alpha])
        group_cov = covered.groupby(group_test.to_numpy()).mean() * 100
        rows.append(
            {
                "quantile": alpha,
                "raw_coverage_pct": raw_cov,
                "conformal_coverage_pct": conf_cov,
                "target_coverage_pct": alpha * 100,
                "pinball_loss_raw": pinball_raw,
                "pinball_loss_conformal": pinball_conf,
                "min_subgroup_coverage_pct": float(group_cov.min()),
                "max_subgroup_coverage_pct": float(group_cov.max()),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["mae_p50"] = mean_absolute_error(actuals_test, test_preds[0.50])
    df.attrs["quantile_crossings"] = int(
        np.sum((test_preds[0.50] > test_preds[0.80]) | (test_preds[0.80] > test_preds[0.90]))
    )
    return df


def _build_synthetic_multicheck_frame(engineered, X_test, test_preds, conformal_test, seed):
    """Fans single-check test cases into small multi-check dossiers.

    The synthetic dataset generates one row per case (matching the
    original prototype's granularity); to exercise the critical-path
    aggregator against something closer to the real multi-component
    dossier structure, each test case is expanded into 1-3 checks that
    share its authority (correlated) with independent jitter.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = engineered.loc[X_test.index, ["authority_id"]].reset_index(drop=True)
    base["case_id"] = [f"CASE-{i:05d}" for i in range(len(base))]
    base["predicted_p50"] = test_preds[0.50]
    base["predicted_p90"] = conformal_test[0.90]

    for i, r in base.iterrows():
        n_checks = rng.integers(1, 4)
        for j in range(n_checks):
            jitter = rng.uniform(0.85, 1.2)
            rows.append(
                {
                    "case_id": r["case_id"],
                    "check_id": f"{r['case_id']}-CHK{j}",
                    "authority_id": r["authority_id"],
                    "predicted_p50": r["predicted_p50"] * jitter,
                    "predicted_p90": r["predicted_p90"] * jitter,
                }
            )
    return pd.DataFrame(rows)


def _print_report(diagnostics, vif_report, model_features, case_level, qhat_reports):
    print("=" * 70)
    print("=== FlowPredict TAT Model -- Strengthened Diagnostics Report ===")
    print("=" * 70)
    print(f"P50 MAE: {diagnostics.attrs['mae_p50']:.2f} days")
    crossings = diagnostics.attrs["quantile_crossings"]
    print(f"Quantile crossing violations (post-clamp): {crossings} (target: 0)")
    print()
    print(diagnostics.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    if not vif_report.empty:
        print("Features dropped for collinearity (VIF-based pruning):")
        print(vif_report.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        print("No features exceeded the VIF collinearity threshold.")
    print(f"\nFinal model feature set ({len(model_features)}): {model_features}")
    print()
    print("Critical-path (multi-check dossier) sample -- copula vs. naive max:")
    sample = case_level.head(5)[["case_id", "case_p50", "case_p80", "case_p90", "naive_max_p90"]]
    print(sample.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    gap = (case_level["case_p90"] - case_level["naive_max_p90"]).mean()
    print(f"\nMean (copula P90 - naive max P90) across {len(case_level)} cases: {gap:+.2f} days")
    print("=" * 70)


if __name__ == "__main__":
    run()
