from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_churn.config import DatasetConfig, SimulationConfig
from customer_churn.dataset import (
    MODEL_FEATURES,
    build_current_scoring_dataset,
    build_modeling_dataset,
)
from customer_churn.modeling import (
    RuleClassifier,
    feature_drift,
    score_with_bundle,
    train_and_evaluate,
)
from customer_churn.simulate import simulate_transactions


@pytest.fixture(scope="module")
def modeling_data() -> pd.DataFrame:
    _, transactions = simulate_transactions(
        SimulationConfig(
            n_customers=180,
            start_date="2023-01-01",
            end_date="2026-06-30",
            seed=12,
        )
    )
    dataset, _ = build_modeling_dataset(transactions, DatasetConfig(), pd.Timestamp("2026-06-30"))
    return dataset


@pytest.fixture(scope="module")
def training_result(modeling_data: pd.DataFrame):
    return train_and_evaluate(modeling_data)


def test_rule_classifier_returns_valid_probabilities() -> None:
    frame = pd.DataFrame({"current_gap_days": [0, 45, 100], "window_progress": [0, 0.5, 2]})
    fixed = RuleClassifier("fixed_90_day").fit(frame, pd.Series([0, 0, 1]))
    personalized = RuleClassifier().fit(frame, pd.Series([0, 0, 1]))
    assert np.allclose(fixed.predict_proba(frame)[:, 1], [0, 0.5, 1])
    assert np.allclose(personalized.predict_proba(frame)[:, 1], [0, 0.5, 1])


def test_unknown_rule_is_rejected() -> None:
    frame = pd.DataFrame({"current_gap_days": [1], "window_progress": [1]})
    with pytest.raises(ValueError, match="Unknown rule"):
        RuleClassifier("unknown").fit(frame, pd.Series([0])).predict_proba(frame)


def test_four_time_periods_are_reported(training_result) -> None:
    split = training_result.metrics["split"]
    assert split["train_rows"] > 0
    assert split["validation_rows"] > 0
    assert split["calibration_rows"] > 0
    assert split["test_rows"] > 0


def test_rules_participate_in_model_selection(training_result) -> None:
    names = set(training_result.comparison["model"])
    assert "Fixed 90-day recency rule" in names
    assert "Personalized window rule" in names
    assert training_result.selected_name in names


def test_test_metrics_include_probability_quality_and_uncertainty(training_result) -> None:
    test = training_result.metrics["test"]
    assert 0 <= test["pr_auc"] <= 1
    assert 0 <= test["brier_score"] <= 1
    assert test["log_loss"] >= 0
    assert "cluster_bootstrap_95" in test
    assert test["cluster_bootstrap_95"]["iterations"] == 200


def test_feature_drift_flags_large_shift(training_result, modeling_data: pd.DataFrame) -> None:
    shifted = modeling_data.head(10).copy()
    shifted[MODEL_FEATURES[0]] += 1_000_000
    drift = feature_drift(shifted, training_result.feature_reference)
    assert drift.loc[drift["feature"] == MODEL_FEATURES[0], "warning"].item()


def test_score_with_bundle_returns_one_probability_per_row(
    training_result, modeling_data: pd.DataFrame
) -> None:
    bundle = {"model": training_result.selected_model, "calibrator": training_result.calibrator}
    probabilities = score_with_bundle(bundle, modeling_data.head(7))
    assert probabilities.shape == (7,)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()


def test_current_scoring_dataset_is_label_free() -> None:
    _, transactions = simulate_transactions(
        SimulationConfig(n_customers=100, start_date="2023-01-01", end_date="2026-06-30", seed=7)
    )
    current, coverage = build_current_scoring_dataset(transactions, DatasetConfig(), "2026-06-30")
    assert "churned_in_personal_window" not in current
    assert current["contact_eligible"].all()
    assert coverage["actionable_customers"] == len(current)


def test_dataset_rejects_transactions_after_boundary() -> None:
    _, transactions = simulate_transactions(
        SimulationConfig(n_customers=20, start_date="2023-01-01", end_date="2025-12-31", seed=2)
    )
    with pytest.raises(ValueError, match="after observation_end"):
        build_modeling_dataset(transactions, DatasetConfig(), "2024-01-01")
