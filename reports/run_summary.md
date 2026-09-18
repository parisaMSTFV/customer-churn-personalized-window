# Reproducible run summary

Synthetic transactions, seed `42`.

## Dataset

- 32,195 eligible historical snapshots
- 41.7% missed their personalized purchase window
- 89.8% of simulated customers had enough retained purchases
- 1,079 label-free operational scores

## Independent time-holdout performance

- Selected candidate: Logistic regression
- PR-AUC: 0.602 (customer-cluster bootstrap 95% CI 0.585–0.618)
- ROC-AUC: 0.678
- Brier score: 0.218
- Log loss: 0.627
- Expected calibration error: 0.014
- Recall in the top 20%: 31.9%
- Lift in the top 20%: 1.59x

## Historical prioritization check

At 20% capacity, risk-only ranking captures 34.7%
of the held-out margin proxy. Probability × expected margin captures
54.1%. The corresponding missed-window
recalls are 29.8% and
25.0%.

The operational score file is label-free and uses HMAC customer keys. These figures measure
historical ranking, not incremental campaign impact. A randomized experiment or uplift model
is required before making a causal retention claim.

Run fingerprint: `cc395adfc4443c47`
