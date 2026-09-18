from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from customer_churn.config import DatasetConfig, SimulationConfig
from customer_churn.dataset import build_modeling_dataset
from customer_churn.modeling import train_and_evaluate
from customer_churn.simulate import simulate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the synthetic benchmark across seeds.")
    parser.add_argument("--customers", type=int, default=700)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--output", type=Path, default=Path("reports/stability.csv"))
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        config = SimulationConfig(n_customers=args.customers, seed=seed)
        _, transactions = simulate_transactions(config)
        dataset, coverage = build_modeling_dataset(transactions, DatasetConfig(), config.end_date)
        result = train_and_evaluate(dataset)
        test = result.metrics["test"]
        rows.append(
            {
                "seed": seed,
                "customers": args.customers,
                "modeling_snapshots": int(coverage["modeling_snapshots"]),
                "selected_model": result.selected_name,
                "pr_auc": test["pr_auc"],
                "roc_auc": test["roc_auc"],
                "brier_score": test["brier_score"],
                "expected_calibration_error": test["expected_calibration_error"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
