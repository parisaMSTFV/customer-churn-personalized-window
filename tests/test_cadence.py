import pandas as pd

from customer_churn.cadence import estimate_cadence
from customer_churn.config import DatasetConfig


def test_stable_40_day_customer_gets_40_day_window() -> None:
    dates = pd.to_datetime(
        ["2025-01-01", "2025-02-10", "2025-03-22", "2025-05-01", "2025-06-10"]
    )
    estimate = estimate_cadence(dates, DatasetConfig())
    assert estimate.expected_gap_days == 40
    assert estimate.personalized_window_days == 40
    assert estimate.gap_dispersion_days == 0


def test_variable_customer_receives_uncertainty_buffer() -> None:
    dates = pd.to_datetime(
        ["2025-01-01", "2025-02-05", "2025-03-17", "2025-05-01", "2025-06-10"]
    )
    estimate = estimate_cadence(dates, DatasetConfig())
    assert estimate.personalized_window_days > estimate.expected_gap_days
