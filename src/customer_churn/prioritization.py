from __future__ import annotations

import numpy as np
import pandas as pd


def build_priority_table(
    scored_rows: pd.DataFrame,
    churn_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Compare pure-risk ranking with probability-weighted customer value."""

    export_columns = [
        "customer_id",
        "score_date",
        "personalized_deadline",
        "personalized_window_days",
        "estimated_margin_next_180d",
    ]
    for optional in ("contact_eligible", "churned_in_personal_window", "heldout_margin_proxy"):
        if optional in scored_rows:
            export_columns.append(optional)
    priority = scored_rows[export_columns].copy()
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
    """Evaluate rankings against held-out outcomes, not the score's own value input."""

    required = {"heldout_margin_proxy", "churned_in_personal_window"}
    if not required.issubset(priority.columns):
        raise ValueError("Historical strategy evaluation requires held-out outcome columns")
    total_value = float(priority["heldout_margin_proxy"].sum())
    total_missed = float(priority["churned_in_personal_window"].sum())
    rows: list[dict[str, float | str]] = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(priority) * fraction)))
        for strategy, ranking_column in [
            ("Risk only", "churn_probability"),
            ("Value at risk", "value_at_risk"),
        ]:
            selected = priority.nlargest(count, ranking_column)
            captured = float(selected["heldout_margin_proxy"].sum())
            missed = float(selected["churned_in_personal_window"].sum())
            rows.append(
                {
                    "capacity": fraction,
                    "strategy": strategy,
                    "selected_customers": float(count),
                    "captured_heldout_margin_proxy": captured,
                    "captured_value_share": captured / max(total_value, 1e-12),
                    "missed_window_recall": missed / max(total_missed, 1.0),
                }
            )
    return pd.DataFrame(rows)
