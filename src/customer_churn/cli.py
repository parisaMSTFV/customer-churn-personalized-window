from __future__ import annotations

import argparse
from pathlib import Path

from customer_churn.pipeline import run_pipeline, score_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict churn against each customer's expected purchase window."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the reproducible end-to-end pipeline.")
    run.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Directory where data, models, and reports are written.",
    )
    run.add_argument("--customers", type=int, default=4_000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--input-transactions",
        type=Path,
        help="Validated .csv or .csv.gz transaction history; bypasses simulation.",
    )
    run.add_argument(
        "--observation-end",
        help="Explicit YYYY-MM-DD data cutoff; required for supplied transaction history.",
    )
    score = subparsers.add_parser("score", help="Create label-free operational scores.")
    score.add_argument("--input-transactions", type=Path, required=True)
    score.add_argument("--observation-end", required=True)
    score.add_argument("--model-bundle", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        metrics = run_pipeline(
            args.project_root,
            args.customers,
            args.seed,
            input_transactions=args.input_transactions,
            observation_end=args.observation_end,
        )
        test = metrics["model"]["test"]
        print(
            "Pipeline complete | "
            f"PR-AUC={test['pr_auc']:.3f} | "
            f"Top-20% lift={test['top_20_percent']['lift']:.2f}x"
        )
    elif args.command == "score":
        metrics = score_pipeline(
            input_transactions=args.input_transactions,
            observation_end=args.observation_end,
            model_bundle=args.model_bundle,
            output_root=args.output_root,
        )
        print(
            "Scoring complete | "
            f"customers={int(metrics['coverage']['actionable_customers'])} | "
            f"drift={metrics['feature_drift']['status']}"
        )


if __name__ == "__main__":
    main()
