# Model card: personalized purchase-window churn

## Intended use

The model ranks customers who have entered the alert portion of their own expected purchase
cycle. It estimates the probability that the next retained purchase will occur after that
customer's personalized deadline. The output supports capacity-limited retention prioritization;
it is not a causal estimate of whether contact will prevent churn.

## Target and eligibility

1. A retained purchase is neither cancelled nor returned.
2. At least four retained purchases and three valid gaps are required.
3. Cadence uses only information available at the snapshot.
4. A robust dispersion buffer extends the median recent gap for irregular customers.
5. The target is one when no retained future purchase occurs on or before the deadline.
6. Deadlines beyond the explicit observation cutoff are excluded.

Current operational scoring is separate. It uses one as-of date, includes only customers between
their alert date and deadline, and exports no target or future-outcome field.

## Training and evaluation

- Candidates: fixed 90-day rule, personalized-window rule, logistic regression, and histogram
  gradient boosting
- Model selection: validation PR-AUC
- Refit: training plus validation periods
- Probability calibration: independent calibration period
- Final metrics: untouched chronological test period
- Uncertainty: 200 customer-cluster bootstrap replicates
- Probability quality: Brier score, log loss, expected calibration error, slope, and intercept
- Operational ranking: precision, recall, and lift at the top 10% and 20%

The historical prioritization comparison evaluates ranking against `missed window × expected
180-day margin`, which requires the held-out future outcome. It does not evaluate a score against
the score itself. It remains a value proxy, not realized incremental profit.

## Model bundle and monitoring

Bundle version `2.0` stores the estimator, calibrator, exact feature order, dataset configuration,
development observation date, software versions, input fingerprint, and development feature
reference. Scoring refuses incompatible bundle or feature versions.

`feature_drift.csv` compares current medians and missing rates with the development reference.
A shift above two robust scale units is marked as a warning. This is a diagnostic only; it is not
an automatic retraining or deployment decision.

## Data and privacy

The committed benchmark is fully synthetic. Supplied history must satisfy
`transaction-history-v2.0` and provide a deployment-owned `CHURN_ID_SALT`. Customer-level CSVs
use HMAC keys and exclude raw identifiers. Source filenames are not copied into provenance.

## Limitations

- Newer customers need a governed cohort-level fallback.
- Median/MAD cadence can react slowly to abrupt behavior changes.
- The simulation simplifies seasonality, availability, marketing contacts, and life events.
- A supplied-history result is population-specific and does not establish transportability.
- Feature-drift thresholds and campaign capacity require production policy ownership.
- Ranking performance does not prove saveability, incremental revenue, or profitability.
