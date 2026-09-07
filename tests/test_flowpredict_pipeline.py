import numpy as np
from src.flowpredict.pipeline import run
from src.flowpredict.survival import ipcw_weights, kaplan_meier
from src.flowpredict.synthetic_data import generate_hierarchical_psv_dataset


def test_synthetic_dataset_has_censored_and_observed_cases():
    cases, authorities, disruptions, closures = generate_hierarchical_psv_dataset(
        n_cases=500, seed=1
    )
    assert (cases["event_observed"] == 1).any()
    assert (cases["event_observed"] == 0).any()
    assert set(cases["country"]).issubset(set(authorities["country"]))


def test_kaplan_meier_survival_is_nonincreasing():
    observed_days = np.array([1, 2, 2, 3, 5, 5, 8], dtype=float)
    event = np.array([1, 1, 0, 1, 1, 0, 1])
    km = kaplan_meier(observed_days, event)
    assert (km["survival"].diff().dropna() <= 1e-9).all()


def test_ipcw_weights_zero_for_censored_positive_for_observed():
    observed_days = np.array([1, 2, 3, 4, 5], dtype=float)
    event = np.array([1, 0, 1, 0, 1])
    w = ipcw_weights(observed_days, event)
    assert (w[event == 0] == 0).all()
    assert (w[event == 1] > 0).all()


def test_pipeline_runs_end_to_end_without_errors():
    result = run(n_cases=1200, seed=7, verbose=False)
    diagnostics = result["diagnostics"]
    assert diagnostics.attrs["quantile_crossings"] == 0
    assert len(result["model_features"]) > 0
    assert result["n_open_cases"] > 0
    assert not result["case_level_critical_path"].empty
    for alpha_cov in diagnostics["conformal_coverage_pct"]:
        assert 0 <= alpha_cov <= 100
