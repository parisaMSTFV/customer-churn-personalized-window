from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from customer_churn.config import DatasetConfig, SimulationConfig
from customer_churn.dataset import (
    MODEL_FEATURES,
    build_current_scoring_dataset,
    build_modeling_dataset,
)
from customer_churn.evaluation import (
    plot_calibration,
    plot_cumulative_gain,
    plot_feature_importance,
    plot_model_comparison,
    plot_personalized_windows,
    plot_priority_comparison,
)
from customer_churn.input_data import load_transactions
from customer_churn.modeling import feature_drift, score_with_bundle, train_and_evaluate
from customer_churn.prioritization import build_priority_table, compare_priority_strategies
from customer_churn.privacy import replace_customer_id, resolve_salt
from customer_churn.simulate import simulate_transactions

MODEL_BUNDLE_VERSION = "2.0"


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values(["customer_id", "order_date", "order_id"]).to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_fingerprint(metrics: dict[str, object]) -> str:
    payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _write_summary(
    path: Path, metrics: dict[str, object], priority_comparison: pd.DataFrame
) -> None:
    test = metrics["model"]["test"]
    coverage = metrics["dataset"]
    top_20 = priority_comparison.loc[priority_comparison["capacity"] == 0.20].set_index("strategy")
    run = metrics["run"]
    if run["data_mode"] == "synthetic_evaluation":
        source_line = f"Synthetic transactions, seed `{run['seed']}`."
        customer_label = "simulated customers"
    else:
        source_line = (
            f"Validated supplied transactions, SHA-256 `{run['input_provenance']['sha256']}`."
        )
        customer_label = "customers in the supplied history"
    bootstrap = test["cluster_bootstrap_95"]["pr_auc"]
    pr_auc_line = (
        f"PR-AUC: {test['pr_auc']:.3f} (customer-cluster bootstrap 95% CI "
        f"{bootstrap['lower_95']:.3f}–{bootstrap['upper_95']:.3f})"
    )
    content = f"""# Reproducible run summary

{source_line}

## Dataset

- {int(coverage["modeling_snapshots"]):,} eligible historical snapshots
- {coverage["positive_rate"]:.1%} missed their personalized purchase window
- {coverage["eligible_customer_share"]:.1%} of {customer_label} had enough retained purchases
- {int(metrics["current_scoring"]["actionable_customers"]):,} label-free operational scores

## Independent time-holdout performance

- Selected candidate: {metrics["model"]["selected_model"]}
- {pr_auc_line}
- ROC-AUC: {test["roc_auc"]:.3f}
- Brier score: {test["brier_score"]:.3f}
- Log loss: {test["log_loss"]:.3f}
- Expected calibration error: {test["expected_calibration_error"]:.3f}
- Recall in the top 20%: {test["top_20_percent"]["recall"]:.1%}
- Lift in the top 20%: {test["top_20_percent"]["lift"]:.2f}x

## Historical prioritization check

At 20% capacity, risk-only ranking captures {top_20.loc["Risk only", "captured_value_share"]:.1%}
of the held-out margin proxy. Probability × expected margin captures
{top_20.loc["Value at risk", "captured_value_share"]:.1%}. The corresponding missed-window
recalls are {top_20.loc["Risk only", "missed_window_recall"]:.1%} and
{top_20.loc["Value at risk", "missed_window_recall"]:.1%}.

The operational score file is label-free and uses HMAC customer keys. These figures measure
historical ranking, not incremental campaign impact. A randomized experiment or uplift model
is required before making a causal retention claim.

Run fingerprint: `{metrics["run_fingerprint"]}`
"""
    path.write_text(content, encoding="utf-8")


def _build_bundle(
    result: object,
    dataset_config: DatasetConfig,
    observation_end: pd.Timestamp,
    input_fingerprint: str,
) -> dict[str, object]:
    return {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "model": result.selected_model,
        "calibrator": result.calibrator,
        "selected_model": result.selected_name,
        "features": MODEL_FEATURES,
        "dataset_config": asdict(dataset_config),
        "development_observation_end": observation_end.date().isoformat(),
        "input_fingerprint": input_fingerprint,
        "feature_reference": result.feature_reference,
        "software_versions": _software_versions(),
    }


def _validate_bundle(bundle: dict[str, object]) -> None:
    if bundle.get("bundle_version") != MODEL_BUNDLE_VERSION:
        raise ValueError(f"Expected model bundle version {MODEL_BUNDLE_VERSION}")
    if bundle.get("features") != MODEL_FEATURES:
        raise ValueError("Model feature schema does not match this package")


def run_pipeline(
    project_root: Path,
    n_customers: int = 4_000,
    seed: int = 42,
    input_transactions: Path | None = None,
    observation_end: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Train/evaluate historically and write a separate label-free current score."""

    project_root = project_root.resolve()
    data_dir = project_root / "data" / "generated"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = project_root / "models"
    for directory in (data_dir, reports_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_config = DatasetConfig()
    if input_transactions is None:
        simulation_config = SimulationConfig(n_customers=n_customers, seed=seed)
        customers, transactions = simulate_transactions(simulation_config)
        boundary = pd.Timestamp(simulation_config.end_date).normalize()
        customers.to_csv(data_dir / "customers.csv.gz", index=False, compression="gzip")
        transactions.to_csv(data_dir / "transactions.csv.gz", index=False, compression="gzip")
        input_fingerprint = _frame_fingerprint(transactions)
        run_metadata: dict[str, object] = {
            "data_mode": "synthetic_evaluation",
            "seed": seed,
            "simulated_customers": n_customers,
            "transactions": int(len(transactions)),
            "observation_end": boundary.date().isoformat(),
        }
        salt = resolve_salt(synthetic=True)
    else:
        if observation_end is None:
            raise ValueError("--observation-end is required with --input-transactions")
        boundary = pd.Timestamp(observation_end).normalize()
        transactions, provenance = load_transactions(input_transactions, observation_end=boundary)
        input_fingerprint = str(provenance["sha256"])
        run_metadata = {
            "data_mode": "supplied_transaction_history",
            "input_provenance": provenance,
            "customers": int(transactions["customer_id"].nunique()),
            "transactions": int(len(transactions)),
            "observation_end": boundary.date().isoformat(),
        }
        salt = resolve_salt(synthetic=False)

    modeling_dataset, coverage = build_modeling_dataset(transactions, dataset_config, boundary)
    current_dataset, current_coverage = build_current_scoring_dataset(
        transactions, dataset_config, boundary
    )
    replace_customer_id(modeling_dataset, salt).to_csv(
        data_dir / "modeling_dataset.csv.gz", index=False, compression="gzip"
    )
    result = train_and_evaluate(modeling_dataset)
    bundle = _build_bundle(result, dataset_config, boundary, input_fingerprint)
    _validate_bundle(bundle)
    joblib.dump(bundle, models_dir / "personalized_churn_model.joblib")

    test_scored = result.test_frame.copy()
    test_scored["_probability"] = result.test_probabilities
    latest_test = (
        test_scored.sort_values("score_date")
        .groupby("customer_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    evaluation_priority = build_priority_table(
        latest_test, latest_test.pop("_probability").to_numpy()
    )
    priority_comparison = compare_priority_strategies(evaluation_priority)

    current_probabilities = score_with_bundle(bundle, current_dataset)
    operational_priority = build_priority_table(current_dataset, current_probabilities)
    drift = feature_drift(current_dataset, result.feature_reference)

    result.comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    result.calibration.to_csv(reports_dir / "calibration.csv", index=False)
    result.feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    replace_customer_id(evaluation_priority.head(100), salt).to_csv(
        reports_dir / "evaluation_priority_sample.csv", index=False
    )
    replace_customer_id(operational_priority, salt).to_csv(
        reports_dir / "operational_scores.csv", index=False
    )
    priority_comparison.to_csv(reports_dir / "priority_strategy_comparison.csv", index=False)
    drift.to_csv(reports_dir / "feature_drift.csv", index=False)

    plot_model_comparison(result.comparison, figures_dir / "model_comparison.png")
    plot_feature_importance(result.feature_importance, figures_dir / "feature_importance.png")
    plot_calibration(result.calibration, figures_dir / "calibration.png")
    plot_cumulative_gain(
        result.test_frame["churned_in_personal_window"],
        result.test_probabilities,
        figures_dir / "cumulative_gain.png",
    )
    plot_personalized_windows(modeling_dataset, figures_dir / "personalized_windows.png")
    plot_priority_comparison(priority_comparison, figures_dir / "priority_comparison.png")

    metrics: dict[str, object] = {
        "run": run_metadata,
        "dataset": coverage,
        "current_scoring": current_coverage,
        "model": result.metrics,
        "feature_drift": {
            "status": "warning" if drift["warning"].any() else "ok",
            "warning_features": drift.loc[drift["warning"], "feature"].tolist(),
            "threshold_robust_units": 2.0,
            "purpose": "diagnostic_only",
        },
        "evaluation_boundary": (
            "Independent historical test ranking; operational scores contain no future label"
        ),
    }
    metrics["run_fingerprint"] = _run_fingerprint(metrics)
    _write_json(reports_dir / "metrics.json", metrics)
    _write_summary(reports_dir / "run_summary.md", metrics, priority_comparison)
    return metrics


def score_pipeline(
    *,
    input_transactions: Path,
    observation_end: str | pd.Timestamp,
    model_bundle: Path,
    output_root: Path,
) -> dict[str, object]:
    """Score supplied current history without constructing or exporting a future label."""

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    boundary = pd.Timestamp(observation_end).normalize()
    transactions, provenance = load_transactions(input_transactions, observation_end=boundary)
    salt = resolve_salt(synthetic=False)
    bundle = joblib.load(model_bundle)
    _validate_bundle(bundle)
    config = DatasetConfig(**bundle["dataset_config"])
    current, coverage = build_current_scoring_dataset(transactions, config, boundary)
    probabilities = score_with_bundle(bundle, current)
    priority = build_priority_table(current, probabilities)
    drift = feature_drift(current, bundle["feature_reference"])
    replace_customer_id(priority, salt).to_csv(output_root / "operational_scores.csv", index=False)
    drift.to_csv(output_root / "feature_drift.csv", index=False)
    metrics: dict[str, object] = {
        "bundle_version": bundle["bundle_version"],
        "selected_model": bundle["selected_model"],
        "input_provenance": provenance,
        "coverage": coverage,
        "feature_drift": {
            "status": "warning" if drift["warning"].any() else "ok",
            "warning_features": drift.loc[drift["warning"], "feature"].tolist(),
            "purpose": "diagnostic_only",
        },
    }
    metrics["scoring_fingerprint"] = _run_fingerprint(metrics)
    _write_json(output_root / "scoring_metrics.json", metrics)
    return metrics
