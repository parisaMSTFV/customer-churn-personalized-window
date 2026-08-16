from __future__ import annotations

import numpy as np
import pandas as pd


def build_priority_table(
    scored_rows: pd.DataFrame,
    churn_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Compare pure-risk ranking with probability-weighted customer value."""

    priority = scored_rows[
        [
            "customer_id",
            "score_date",
            "personalized_deadline",
            "personalized_window_days",
            "estimated_margin_next_180d",
            "churned_in_personal_window",
        ]
    ].copy()
    priority["churn_probability"] = churn_probabilities
    priority["value_at_risk"] = (
        priority["churn_probability"] * priority["estimated_margin_next_180d"]
    )
    priority["risk_rank"] = priority["churn_probability"].rank(method="first", ascending=False)
    priority["value_at_risk_rank"] = priority["value_at_risk"].rank(method="first", ascending=False)
    return priority.sort_values("value_at_risk", ascending=False).reset_index(drop=True)


def compare_priority_strategies(
    priority: pd.DataFrame,
    fractions: tuple[float, ...] = (0.10, 0.20),
) -> pd.DataFrame:
    """Measure how much modeled value at risk each capacity-limited ranking captures."""

    total_value_at_risk = float(priority["value_at_risk"].sum())
    rows: list[dict[str, float | str]] = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(priority) * fraction)))
        for strategy, ranking_column in [
            ("Risk only", "churn_probability"),
            ("Value at risk", "value_at_risk"),
        ]:
            selected = priority.nlargest(count, ranking_column)
            captured = float(selected["value_at_risk"].sum())
            rows.append(
                {
                    "capacity": fraction,
                    "strategy": strategy,
                    "selected_customers": float(count),
                    "captured_value_at_risk": captured,
                    "captured_value_share": captured / max(total_value_at_risk, 1e-12),
                }
            )
    return pd.DataFrame(rows)
