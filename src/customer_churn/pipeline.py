from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from customer_churn.config import DatasetConfig, SimulationConfig
from customer_churn.dataset import build_modeling_dataset
from customer_churn.evaluation import (
    plot_calibration,
    plot_cumulative_gain,
    plot_feature_importance,
    plot_model_comparison,
    plot_personalized_windows,
    plot_priority_comparison,
)
from customer_churn.modeling import train_and_evaluate
from customer_churn.prioritization import (
    build_priority_table,
    compare_priority_strategies,
)
from customer_churn.simulate import simulate_transactions


def _write_summary(
    path: Path,
    metrics: dict[str, object],
    priority_comparison: pd.DataFrame,
) -> None:
    test = metrics["model"]["test"]
    coverage = metrics["dataset"]
    top_20 = priority_comparison.loc[priority_comparison["capacity"] == 0.20].set_index(
        "strategy"
    )
    content = f"""# Reproducible run summary

This report was generated from synthetic transactions with seed `{metrics["run"]["seed"]}`.

## Dataset

- {int(coverage["modeling_snapshots"]):,} eligible customer snapshots
- {coverage["positive_rate"]:.1%} missed their personalized purchase window
- {coverage["eligible_customer_share"]:.1%} of simulated customers had enough history
  for a personalized cadence

## Time-holdout performance

- Selected model: {metrics["model"]["selected_model"]}
- PR-AUC: {test["pr_auc"]:.3f}
- ROC-AUC: {test["roc_auc"]:.3f}
- Brier score: {test["brier_score"]:.3f}
- Recall in the top 20%: {test["top_20_percent"]["recall"]:.1%}
- Lift in the top 20%: {test["top_20_percent"]["lift"]:.2f}x

## Capacity-limited prioritization

At 20% campaign capacity, ranking by churn probability alone captures
{top_20.loc["Risk only", "captured_value_share"]:.1%} of modeled value at risk.
Ranking by probability × expected margin captures
{top_20.loc["Value at risk", "captured_value_share"]:.1%}.

These figures describe ranking performance on synthetic data. They do not estimate
incremental campaign impact; that requires a randomized experiment or uplift model.
"""
    path.write_text(content, encoding="utf-8")


def run_pipeline(
    project_root: Path,
    n_customers: int = 4_000,
    seed: int = 42,
) -> dict[str, object]:
    """Run data generation, feature engineering, modeling, and reporting."""

    project_root = project_root.resolve()
    data_dir = project_root / "data" / "generated"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = project_root / "models"
    for directory in [data_dir, reports_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    simulation_config = SimulationConfig(n_customers=n_customers, seed=seed)
    dataset_config = DatasetConfig()
    customers, transactions = simulate_transactions(simulation_config)
    customers.to_csv(data_dir / "customers.csv.gz", index=False, compression="gzip")
    transactions.to_csv(data_dir / "transactions.csv.gz", index=False, compression="gzip")

    modeling_dataset, coverage = build_modeling_dataset(transactions, dataset_config)
    modeling_dataset.to_csv(
        data_dir / "modeling_dataset.csv.gz", index=False, compression="gzip"
    )
    result = train_and_evaluate(modeling_dataset)
    joblib.dump(
        {
            "model": result.selected_model,
            "calibrator": result.calibrator,
        },
        models_dir / "personalized_churn_model.joblib",
    )

    test_with_probabilities = result.test_frame.copy()
    test_with_probabilities["_probability"] = result.test_probabilities
    latest_scored = (
        test_with_probabilities.sort_values("score_date")
        .groupby("customer_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    priority = build_priority_table(
        latest_scored,
        latest_scored.pop("_probability").to_numpy(),
    )
    priority_comparison = compare_priority_strategies(priority)

    result.comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    result.calibration.to_csv(reports_dir / "calibration.csv", index=False)
    result.feature_importance.to_csv(
        reports_dir / "feature_importance.csv", index=False
    )
    priority.head(100).to_csv(reports_dir / "priority_sample.csv", index=False)
    priority_comparison.to_csv(
        reports_dir / "priority_strategy_comparison.csv", index=False
    )

    plot_model_comparison(result.comparison, figures_dir / "model_comparison.png")
    plot_feature_importance(
        result.feature_importance, figures_dir / "feature_importance.png"
    )
    plot_calibration(result.calibration, figures_dir / "calibration.png")
    plot_cumulative_gain(
        result.test_frame["churned_in_personal_window"],
        result.test_probabilities,
        figures_dir / "cumulative_gain.png",
    )
    plot_personalized_windows(
        modeling_dataset, figures_dir / "personalized_windows.png"
    )
    plot_priority_comparison(
        priority_comparison, figures_dir / "priority_comparison.png"
    )

    metrics: dict[str, object] = {
        "run": {
            "seed": seed,
            "simulated_customers": n_customers,
            "synthetic_transactions": int(len(transactions)),
        },
        "dataset": coverage,
        "model": result.metrics,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_summary(reports_dir / "run_summary.md", metrics, priority_comparison)
    return metrics
