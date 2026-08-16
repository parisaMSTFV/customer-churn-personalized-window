# Reproducible run summary

This report was generated from synthetic transactions with seed `42`.

## Dataset

- 33,486 eligible customer snapshots
- 41.4% missed their personalized purchase window
- 90.6% of simulated customers had enough history
  for a personalized cadence

## Time-holdout performance

- Selected model: Logistic regression
- PR-AUC: 0.602
- ROC-AUC: 0.685
- Brier score: 0.217
- Recall in the top 20%: 32.0%
- Lift in the top 20%: 1.60x

## Capacity-limited prioritization

At 20% campaign capacity, ranking by churn probability alone captures
35.6% of modeled value at risk.
Ranking by probability × expected margin captures
54.3%.

These figures describe time-holdout predictive ranking for this input. They do not estimate
incremental campaign impact; that requires a randomized experiment or uplift model. Customer
identifiers remain in derived artifacts and require appropriate governance.
