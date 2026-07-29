from __future__ import annotations

import argparse
from pathlib import Path

from customer_churn.pipeline import run_pipeline


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
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        metrics = run_pipeline(args.project_root, args.customers, args.seed)
        test = metrics["model"]["test"]
        print(
            "Pipeline complete | "
            f"PR-AUC={test['pr_auc']:.3f} | "
            f"Top-20% lift={test['top_20_percent']['lift']:.2f}x"
        )


if __name__ == "__main__":
    main()
