import numpy as np
import pandas as pd

from customer_churn.prioritization import (
    build_priority_table,
    compare_priority_strategies,
)


def test_value_at_risk_can_change_retention_priority() -> None:
    scored = pd.DataFrame(
        {
            "customer_id": ["high-risk-low-value", "lower-risk-high-value"],
            "score_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "personalized_deadline": pd.to_datetime(["2026-01-10", "2026-01-10"]),
            "personalized_window_days": [20, 20],
            "estimated_margin_next_180d": [10.0, 100.0],
            "churned_in_personal_window": [1, 0],
            "heldout_margin_proxy": [25.0, 100.0],
        }
    )
    priority = build_priority_table(scored, np.array([0.90, 0.50]))
    assert priority.iloc[0]["customer_id"] == "lower-risk-high-value"

    comparison = compare_priority_strategies(priority, fractions=(0.50,))
    value_capture = comparison.set_index("strategy")["captured_value_share"]
    assert value_capture["Value at risk"] > value_capture["Risk only"]
