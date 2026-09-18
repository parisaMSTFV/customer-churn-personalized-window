import pandas as pd

from customer_churn.cadence import estimate_cadence
from customer_churn.config import DatasetConfig
from customer_churn.dataset import MODEL_FEATURES, customer_features


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(1, 7)],
            "order_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-10",
                    "2025-03-22",
                    "2025-05-01",
                    "2025-06-10",
                    "2025-08-01",
                ]
            ),
            "gross_value": [100, 110, 120, 130, 140, 9999],
            "contribution_margin": [20, 21, 22, 23, 24, 9999],
            "discount_pct": [0.1] * 6,
            "delivery_delay_days": [0] * 6,
            "is_cancelled": [0] * 6,
            "is_returned": [0] * 6,
            "category_id": [1, 1, 2, 2, 3, 9],
        }
    )


def test_future_order_does_not_change_snapshot_features() -> None:
    events = _events()
    config = DatasetConfig()
    history_dates = events.loc[:4, "order_date"]
    cadence = estimate_cadence(history_dates, config)
    score_date = pd.Timestamp("2025-07-01")
    with_future = customer_features(events, score_date, pd.Timestamp("2025-06-10"), cadence, 180)
    without_future = customer_features(
        events.iloc[:5], score_date, pd.Timestamp("2025-06-10"), cadence, 180
    )
    assert with_future == without_future


def test_model_features_exclude_target_and_future_dates() -> None:
    forbidden = {"churned_in_personal_window", "personalized_deadline", "next_order_date"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_returned_order_is_not_counted_as_successful() -> None:
    events = _events().iloc[:5].copy()
    events.loc[4, "is_returned"] = 1
    cadence = estimate_cadence(events.loc[:3, "order_date"], DatasetConfig())
    features = customer_features(
        events,
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-05-01"),
        cadence,
        180,
    )
    assert features["successful_orders_180d"] == 3
    assert features["failure_rate_180d"] > 0
