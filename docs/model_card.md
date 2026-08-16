# Model card: personalized purchase-window churn

## Intended use

The model ranks customers who have entered the alert portion of their own expected purchase
cycle. It estimates the probability that the next successful purchase will occur after that
customer's personalized deadline.

The output supports retention prioritization under limited campaign capacity. It is not a
causal estimate of whether an intervention will prevent churn.

## Target definition

At each scoring snapshot:

1. Estimate expected cadence from successful purchases available before the snapshot.
2. Add a robust uncertainty buffer based on recent interpurchase-gap dispersion.
3. Define the deadline as the last successful purchase date plus that personalized window.
4. Set the target to one when no successful future purchase occurs on or before the deadline.

Snapshots whose deadlines extend beyond the observation period are removed to avoid treating
right-censored customers as churned.

## Training and evaluation

- Models: logistic regression and histogram gradient boosting
- Baselines: fixed 90-day recency and a personalized window-progress rule
- Validation: chronological train, calibration, and test periods
- Primary metric: PR-AUC
- Operational metrics: lift and recall at the top 10% and 20%
- Probability quality: Brier score and calibration curve

## Data and provenance

The checked-in benchmark is synthetic. The generator creates heterogeneous customer cadence,
order value, promotion affinity, experience failures, and purchase slowdown. It does not place
a churn label directly on a customer; the target is calculated from the simulated event sequence.

The pipeline also accepts validated transaction histories under `transaction-history-v1.0`.
Supplied runs record a content checksum and evaluate a chronologically held-out period; they do
not reuse the checked-in synthetic performance claim. Derived artifacts retain customer IDs.

## Limitations

- Customers need at least four successful historical orders. A segment-level or survival-model
  fallback is required for newer customers.
- A median-and-MAD cadence estimate may react slowly to abrupt life-cycle changes.
- Repeated seasonality, stock availability, and marketing contacts are simplified.
- A supplied-history run is population-specific and does not by itself establish transportability.
- Good ranking performance does not prove that a retention action is incremental or profitable.
- Before production use, fairness, stability, drift, contact policy, and experiment design must
  be reviewed with real, governed data.
