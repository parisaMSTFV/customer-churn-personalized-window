from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from customer_churn.config import DatasetConfig


@dataclass(frozen=True)
class CadenceEstimate:
    expected_gap_days: float
    gap_dispersion_days: float
    personalized_window_days: int
    cadence_cv: float
    cadence_trend_ratio: float


def estimate_cadence(
    order_dates: pd.Series | list[pd.Timestamp],
    config: DatasetConfig,
) -> CadenceEstimate:
    """Estimate a robust customer-specific purchase window from past orders only."""

    dates = pd.Series(pd.to_datetime(order_dates)).drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna().astype(float)
    gaps = gaps[(gaps >= 1) & (gaps <= 365)]
    if len(gaps) < config.min_successful_orders - 1:
        raise ValueError("Not enough historical purchase gaps for a personalized window.")

    recent = gaps.tail(config.recent_gaps)
    expected_gap = float(np.median(recent))
    mad = float(np.median(np.abs(recent - expected_gap)))
    robust_dispersion = 1.4826 * mad
    raw_window = expected_gap + config.dispersion_buffer * robust_dispersion
    window = int(np.clip(np.ceil(raw_window), config.min_window_days, config.max_window_days))

    if len(gaps) >= 5:
        recent_median = float(np.median(gaps.tail(3)))
        previous_median = float(np.median(gaps.iloc[-6:-3]))
        trend_ratio = recent_median / max(previous_median, 1.0)
    else:
        trend_ratio = 1.0

    return CadenceEstimate(
        expected_gap_days=expected_gap,
        gap_dispersion_days=robust_dispersion,
        personalized_window_days=window,
        cadence_cv=robust_dispersion / max(expected_gap, 1.0),
        cadence_trend_ratio=trend_ratio,
    )


def first_scoring_date(
    alert_start: pd.Timestamp,
    origin: pd.Timestamp,
    frequency_days: int,
) -> pd.Timestamp:
    """Return the first scoring run on or after a customer's alert date."""

    elapsed_days = max(0, (alert_start.normalize() - origin.normalize()).days)
    periods = int(np.ceil(elapsed_days / frequency_days))
    return origin.normalize() + pd.Timedelta(days=periods * frequency_days)
