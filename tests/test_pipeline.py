from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from customer_churn.pipeline import run_pipeline, score_pipeline


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("pipeline")
    metrics = run_pipeline(root, n_customers=180, seed=9)
    return root, metrics


def test_pipeline_writes_versioned_model_bundle(completed_run) -> None:
    root, _ = completed_run
    bundle = joblib.load(root / "models" / "personalized_churn_model.joblib")
    assert bundle["bundle_version"] == "2.0"
    assert bundle["features"]
    assert bundle["feature_reference"].shape[0] == len(bundle["features"])
    assert "software_versions" in bundle


def test_pipeline_separates_evaluation_and_operational_exports(completed_run) -> None:
    root, _ = completed_run
    operational = pd.read_csv(root / "reports" / "operational_scores.csv")
    evaluation = pd.read_csv(root / "reports" / "evaluation_priority_sample.csv")
    assert "churned_in_personal_window" not in operational
    assert "customer_id" not in operational
    assert "customer_key" in operational
    assert "churned_in_personal_window" in evaluation


def test_pipeline_records_deterministic_fingerprint(completed_run) -> None:
    root, metrics = completed_run
    disk = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert disk["run_fingerprint"] == metrics["run_fingerprint"]
    assert len(metrics["run_fingerprint"]) == 16


def test_pipeline_writes_diagnostic_drift_report(completed_run) -> None:
    root, metrics = completed_run
    drift = pd.read_csv(root / "reports" / "feature_drift.csv")
    assert set(["feature", "warning"]).issubset(drift.columns)
    assert metrics["feature_drift"]["purpose"] == "diagnostic_only"


def test_supplied_scoring_requires_salt(
    completed_run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _ = completed_run
    monkeypatch.delenv("CHURN_ID_SALT", raising=False)
    with pytest.raises(ValueError, match="CHURN_ID_SALT"):
        score_pipeline(
            input_transactions=root / "data" / "generated" / "transactions.csv.gz",
            observation_end="2026-06-30",
            model_bundle=root / "models" / "personalized_churn_model.joblib",
            output_root=tmp_path,
        )


def test_score_pipeline_exports_no_truth_or_raw_identifier(
    completed_run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _ = completed_run
    monkeypatch.setenv("CHURN_ID_SALT", "0123456789abcdef")
    metrics = score_pipeline(
        input_transactions=root / "data" / "generated" / "transactions.csv.gz",
        observation_end="2026-06-30",
        model_bundle=root / "models" / "personalized_churn_model.joblib",
        output_root=tmp_path,
    )
    output = pd.read_csv(tmp_path / "operational_scores.csv")
    assert "customer_id" not in output
    assert "churned_in_personal_window" not in output
    assert metrics["coverage"]["actionable_customers"] == len(output)
    assert (tmp_path / "scoring_metrics.json").exists()


def test_supplied_training_requires_observation_end(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="observation-end"):
        run_pipeline(tmp_path, input_transactions=tmp_path / "transactions.csv")
