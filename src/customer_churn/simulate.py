from __future__ import annotations

import math

import numpy as np
import pandas as pd

from customer_churn.config import SimulationConfig

SEGMENTS = {
    "frequent": (0.24, 14.0, 0.22),
    "regular": (0.51, 34.0, 0.25),
    "occasional": (0.25, 68.0, 0.28),
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _customer_table(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    segment_names = list(SEGMENTS)
    probabilities = [SEGMENTS[name][0] for name in segment_names]
    segments = rng.choice(segment_names, size=config.n_customers, p=probabilities)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    acquisition_horizon_days = max(31, (end - start).days - 30)

    rows: list[dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        _, typical_gap, gap_sigma = SEGMENTS[str(segment)]
        base_gap = float(rng.lognormal(np.log(typical_gap), gap_sigma))
        rows.append(
            {
                "customer_id": f"C{index:06d}",
                "signup_date": start
                + pd.Timedelta(days=int(rng.integers(0, acquisition_horizon_days))),
                "cadence_segment": str(segment),
                "base_gap_days": float(np.clip(base_gap, 5, 110)),
                "base_order_value": float(rng.lognormal(np.log(85), 0.55)),
                "promo_affinity": float(rng.beta(2.2, 3.4)),
                "experience_risk": float(rng.beta(1.5, 8.0)),
                "cadence_drift": float(np.clip(rng.normal(0.006, 0.012), -0.02, 0.04)),
                "acquisition_channel": str(
                    rng.choice(
                        ["organic", "paid", "referral", "partner"],
                        p=[0.42, 0.31, 0.17, 0.10],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def simulate_transactions(
    config: SimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate customer profiles and order events without preassigned churn labels.

    Attrition emerges through a sequential purchase process. Poor experiences, promotion
    dependence, and slowing cadence affect future gaps and the chance that purchasing stops.
    The modeling label is created later from observed future transactions.
    """

    rng = np.random.default_rng(config.seed)
    customers = _customer_table(config, rng)
    end_date = pd.Timestamp(config.end_date)
    orders: list[dict[str, object]] = []
    order_number = 1

    for customer in customers.itertuples(index=False):
        current_date = pd.Timestamp(customer.signup_date) + pd.Timedelta(
            days=int(rng.integers(0, max(2, min(31, round(customer.base_gap_days)))))
        )
        recent_bad_experience = 0.0
        customer_categories = rng.choice(
            np.arange(1, 13),
            size=int(rng.integers(2, 6)),
            replace=False,
        )
        event_index = 0

        while current_date <= end_date and event_index < 120:
            seasonal_multiplier = 1.12 if current_date.month in {11, 12} else 1.0
            gross_value = float(
                rng.lognormal(np.log(customer.base_order_value * seasonal_multiplier), 0.38)
            )
            discount_pct = float(
                np.clip(rng.beta(1.4 + 4.0 * customer.promo_affinity, 6.5) * 0.55, 0, 0.60)
            )

            cancel_probability = 0.008 + 0.055 * customer.experience_risk
            return_probability = 0.018 + 0.065 * customer.experience_risk
            delay_probability = 0.06 + 0.20 * customer.experience_risk
            is_cancelled = bool(rng.random() < cancel_probability)
            is_returned = bool((not is_cancelled) and rng.random() < return_probability)
            is_delayed = bool(rng.random() < delay_probability)
            delivery_delay_days = (
                int(rng.integers(3, 11)) if is_delayed else int(rng.integers(0, 3))
            )
            is_bad_experience = float(is_cancelled or is_returned or delivery_delay_days >= 3)

            margin_rate = 0.24 - 0.32 * discount_pct + float(rng.normal(0, 0.025))
            contribution_margin = 0.0 if is_cancelled else gross_value * margin_rate
            orders.append(
                {
                    "order_id": f"O{order_number:08d}",
                    "customer_id": customer.customer_id,
                    "order_date": current_date,
                    "gross_value": round(gross_value, 2),
                    "contribution_margin": round(contribution_margin, 2),
                    "discount_pct": round(discount_pct, 4),
                    "delivery_delay_days": delivery_delay_days,
                    "is_cancelled": int(is_cancelled),
                    "is_returned": int(is_returned),
                    "category_id": int(rng.choice(customer_categories)),
                }
            )
            order_number += 1
            event_index += 1

            if not is_cancelled and event_index >= 4:
                attrition_logit = (
                    -5.8
                    + 1.45 * is_bad_experience
                    + 1.00 * recent_bad_experience
                    + 0.85 * customer.promo_affinity
                    + 0.55 * (customer.base_gap_days / 60.0)
                    + 0.25 * max(customer.cadence_drift, 0) * event_index
                )
                if rng.random() < _sigmoid(attrition_logit):
                    break

            drift_multiplier = float(np.clip(1.0 + customer.cadence_drift * event_index, 0.65, 2.2))
            experience_multiplier = 1.0 + 0.55 * is_bad_experience + 0.25 * recent_bad_experience
            random_multiplier = float(rng.lognormal(0, 0.27))
            next_gap = customer.base_gap_days * drift_multiplier * experience_multiplier
            next_gap *= random_multiplier
            if is_cancelled:
                next_gap *= 0.35
            current_date += pd.Timedelta(days=max(1, int(round(next_gap))))
            recent_bad_experience = 0.65 * recent_bad_experience + is_bad_experience

    transactions = pd.DataFrame(orders).sort_values(["customer_id", "order_date", "order_id"])
    return customers, transactions.reset_index(drop=True)
