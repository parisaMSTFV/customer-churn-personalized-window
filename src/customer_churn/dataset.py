from __future__ import annotations

import math

import numpy as np
import pandas as pd

from customer_churn.cadence import CadenceEstimate, estimate_cadence, first_scoring_date
from customer_churn.config import DatasetConfig

MODEL_FEATURES = [
    "expected_gap_days",
    "gap_dispersion_days",
    "personalized_window_days",
    "cadence_cv",
    "cadence_trend_ratio",
    "current_gap_days",
    "gap_ratio",
    "window_progress",
    "successful_orders_180d",
    "successful_orders_365d",
    "average_order_value_180d",
    "margin_180d",
    "average_discount_180d",
    "failure_rate_180d",
    "category_diversity_180d",
    "customer_tenure_days",
]


def customer_features(
    events: pd.DataFrame,
    score_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    cadence: CadenceEstimate,
    lookback_days: int,
) -> dict[str, float]:
    """Create features using events available on or before the scoring date."""

    score_date = pd.Timestamp(score_date)
    visible = events.loc[pd.to_datetime(events["order_date"]) <= score_date].copy()
    if visible.empty:
        raise ValueError("No customer history is visible at the scoring date.")

    visible["order_date"] = pd.to_datetime(visible["order_date"])
    recent = visible.loc[
        visible["order_date"] >= score_date - pd.to_timedelta(lookback_days, unit="D")
    ]
    annual = visible.loc[
        visible["order_date"] >= score_date - pd.to_timedelta(365, unit="D")
    ]
    recent_success = recent.loc[recent["is_cancelled"] == 0]
    annual_success = annual.loc[annual["is_cancelled"] == 0]
    failed = (
        (recent["is_cancelled"] == 1)
        | (recent["is_returned"] == 1)
        | (recent["delivery_delay_days"] >= 3)
    )
    current_gap = float((score_date - pd.Timestamp(anchor_date)).days)

    return {
        "expected_gap_days": cadence.expected_gap_days,
        "gap_dispersion_days": cadence.gap_dispersion_days,
        "personalized_window_days": float(cadence.personalized_window_days),
        "cadence_cv": cadence.cadence_cv,
        "cadence_trend_ratio": cadence.cadence_trend_ratio,
        "current_gap_days": current_gap,
        "gap_ratio": current_gap / max(cadence.expected_gap_days, 1.0),
        "window_progress": current_gap / max(cadence.personalized_window_days, 1),
        "successful_orders_180d": float(len(recent_success)),
        "successful_orders_365d": float(len(annual_success)),
        "average_order_value_180d": float(recent_success["gross_value"].mean())
        if not recent_success.empty
        else 0.0,
        "margin_180d": float(recent_success["contribution_margin"].sum()),
        "average_discount_180d": float(recent_success["discount_pct"].mean())
        if not recent_success.empty
        else 0.0,
        "failure_rate_180d": float(failed.mean()) if not recent.empty else 0.0,
        "category_diversity_180d": float(recent_success["category_id"].nunique()),
        "customer_tenure_days": float(
            (score_date - pd.Timestamp(visible["order_date"].min())).days
        ),
    }


def _array_features(
    event_dates: np.ndarray,
    gross_value: np.ndarray,
    contribution_margin: np.ndarray,
    discount_pct: np.ndarray,
    delivery_delay_days: np.ndarray,
    is_cancelled: np.ndarray,
    is_returned: np.ndarray,
    category_id: np.ndarray,
    score_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    cadence: CadenceEstimate,
    lookback_days: int,
) -> dict[str, float]:
    """Fast point-in-time feature calculation for the dataset builder."""

    score_value = np.datetime64(score_date, "ns")
    visible_end = int(np.searchsorted(event_dates, score_value, side="right"))
    visible_dates = event_dates[:visible_end]
    recent_start_value = np.datetime64(
        score_date - pd.Timedelta(days=lookback_days), "ns"
    )
    annual_start_value = np.datetime64(score_date - pd.Timedelta(days=365), "ns")
    recent_start = int(
        np.searchsorted(visible_dates, recent_start_value, side="left")
    )
    annual_start = int(
        np.searchsorted(visible_dates, annual_start_value, side="left")
    )

    recent_slice = slice(recent_start, visible_end)
    annual_slice = slice(annual_start, visible_end)
    recent_success = is_cancelled[recent_slice] == 0
    annual_success = is_cancelled[annual_slice] == 0
    recent_values = gross_value[recent_slice][recent_success]
    recent_margins = contribution_margin[recent_slice][recent_success]
    recent_discounts = discount_pct[recent_slice][recent_success]
    recent_categories = category_id[recent_slice][recent_success]
    recent_failures = (
        (is_cancelled[recent_slice] == 1)
        | (is_returned[recent_slice] == 1)
        | (delivery_delay_days[recent_slice] >= 3)
    )
    current_gap = float((score_date - anchor_date).days)
    first_visible_date = pd.Timestamp(visible_dates[0])

    return {
        "expected_gap_days": cadence.expected_gap_days,
        "gap_dispersion_days": cadence.gap_dispersion_days,
        "personalized_window_days": float(cadence.personalized_window_days),
        "cadence_cv": cadence.cadence_cv,
        "cadence_trend_ratio": cadence.cadence_trend_ratio,
        "current_gap_days": current_gap,
        "gap_ratio": current_gap / max(cadence.expected_gap_days, 1.0),
        "window_progress": current_gap / max(cadence.personalized_window_days, 1),
        "successful_orders_180d": float(recent_success.sum()),
        "successful_orders_365d": float(annual_success.sum()),
        "average_order_value_180d": float(recent_values.mean())
        if len(recent_values)
        else 0.0,
        "margin_180d": float(recent_margins.sum()),
        "average_discount_180d": float(recent_discounts.mean())
        if len(recent_discounts)
        else 0.0,
        "failure_rate_180d": float(recent_failures.mean())
        if len(recent_failures)
        else 0.0,
        "category_diversity_180d": float(np.unique(recent_categories).size),
        "customer_tenure_days": float((score_date - first_visible_date).days),
    }


def build_modeling_dataset(
    transactions: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create one actionable snapshot per eligible interpurchase spell.

    A snapshot is scheduled after the customer reaches the alert fraction of their
    expected cadence and before their personalized deadline. The future purchase is
    used only to create the target.
    """

    transactions = transactions.copy()
    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    transactions = transactions.sort_values(["customer_id", "order_date", "order_id"])
    observation_end = pd.Timestamp(transactions["order_date"].max()).normalize()
    origin = pd.Timestamp(transactions["order_date"].min()).normalize()
    rows: list[dict[str, object]] = []
    eligible_customers: set[str] = set()

    for customer_id, events in transactions.groupby("customer_id", sort=False):
        events = events.sort_values(["order_date", "order_id"]).reset_index(drop=True)
        successful = events.loc[events["is_cancelled"] == 0].reset_index(drop=True)
        if len(successful) < config.min_successful_orders:
            continue
        eligible_customers.add(str(customer_id))
        event_dates = events["order_date"].to_numpy(dtype="datetime64[ns]")
        gross_value = events["gross_value"].to_numpy(dtype=float)
        contribution_margin = events["contribution_margin"].to_numpy(dtype=float)
        discount_pct = events["discount_pct"].to_numpy(dtype=float)
        delivery_delay_days = events["delivery_delay_days"].to_numpy(dtype=int)
        is_cancelled = events["is_cancelled"].to_numpy(dtype=int)
        is_returned = events["is_returned"].to_numpy(dtype=int)
        category_id = events["category_id"].to_numpy(dtype=int)

        for anchor_index in range(config.min_successful_orders - 1, len(successful)):
            history = successful.iloc[: anchor_index + 1]
            cadence = estimate_cadence(history["order_date"], config)
            anchor_date = pd.Timestamp(history.iloc[-1]["order_date"]).normalize()
            alert_days = max(1, math.ceil(config.alert_fraction * cadence.expected_gap_days))
            alert_start = anchor_date + pd.Timedelta(days=alert_days)
            score_date = first_scoring_date(
                alert_start=alert_start,
                origin=origin,
                frequency_days=config.score_frequency_days,
            )
            deadline = anchor_date + pd.Timedelta(days=cadence.personalized_window_days)

            if score_date >= deadline or deadline > observation_end:
                continue

            next_date = (
                pd.Timestamp(successful.iloc[anchor_index + 1]["order_date"]).normalize()
                if anchor_index + 1 < len(successful)
                else pd.NaT
            )
            if pd.notna(next_date) and next_date <= score_date:
                continue

            feature_values = _array_features(
                event_dates=event_dates,
                gross_value=gross_value,
                contribution_margin=contribution_margin,
                discount_pct=discount_pct,
                delivery_delay_days=delivery_delay_days,
                is_cancelled=is_cancelled,
                is_returned=is_returned,
                category_id=category_id,
                score_date=score_date,
                anchor_date=anchor_date,
                cadence=cadence,
                lookback_days=config.feature_lookback_days,
            )
            average_margin = (
                feature_values["margin_180d"]
                / max(feature_values["successful_orders_180d"], 1.0)
            )
            estimated_margin_next_180d = max(0.0, average_margin) * (
                180.0 / max(cadence.expected_gap_days, 1.0)
            )
            rows.append(
                {
                    "customer_id": str(customer_id),
                    "score_date": score_date,
                    "anchor_order_date": anchor_date,
                    "personalized_deadline": deadline,
                    "churned_in_personal_window": int(
                        pd.isna(next_date) or next_date > deadline
                    ),
                    "estimated_margin_next_180d": estimated_margin_next_180d,
                    **feature_values,
                }
            )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("No eligible snapshots were created.")
    dataset = dataset.sort_values(["score_date", "customer_id"]).reset_index(drop=True)
    coverage = {
        "total_customers": float(transactions["customer_id"].nunique()),
        "eligible_customers": float(len(eligible_customers)),
        "eligible_customer_share": float(
            len(eligible_customers) / transactions["customer_id"].nunique()
        ),
        "modeling_snapshots": float(len(dataset)),
        "positive_rate": float(dataset["churned_in_personal_window"].mean()),
    }
    return dataset, coverage
