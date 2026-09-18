from __future__ import annotations

import sys
from pathlib import Path

from customer_churn import cli


def test_run_cli_forwards_observation_end(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(project_root, customers, seed, **kwargs):
        captured.update(kwargs)
        return {"model": {"test": {"pr_auc": 0.5, "top_20_percent": {"lift": 1.2}}}}

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "churn-pipeline",
            "run",
            "--input-transactions",
            "input.csv",
            "--observation-end",
            "2026-01-01",
        ],
    )
    cli.main()
    assert captured["observation_end"] == "2026-01-01"
    assert "Pipeline complete" in capsys.readouterr().out


def test_score_cli_forwards_paths(monkeypatch, capsys) -> None:
    captured = {}

    def fake_score(**kwargs):
        captured.update(kwargs)
        return {
            "coverage": {"actionable_customers": 3},
            "feature_drift": {"status": "ok"},
        }

    monkeypatch.setattr(cli, "score_pipeline", fake_score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "churn-pipeline",
            "score",
            "--input-transactions",
            "input.csv",
            "--observation-end",
            "2026-01-01",
            "--model-bundle",
            "model.joblib",
            "--output-root",
            "output",
        ],
    )
    cli.main()
    assert captured["model_bundle"] == Path("model.joblib")
    assert "customers=3" in capsys.readouterr().out
