# Transaction-history input contract

`churn-pipeline run --input-transactions` and `churn-pipeline score` accept UTF-8 `.csv`
or `.csv.gz` files under contract `transaction-history-v2.0`.

## Required command inputs

- `--observation-end YYYY-MM-DD` is mandatory for supplied history. It is the governed
  extraction cutoff, not a date inferred from the final transaction.
- `CHURN_ID_SALT` must contain at least 16 characters. The deployment owner supplies and
  protects it; it is never written to an artifact.
- The input must contain exactly the columns below, in this order. Extra columns are rejected.

| Field | Type | Validation |
|---|---|---|
| `order_id` | string | Non-empty, unique, at most 128 characters |
| `customer_id` | string | Non-empty, at most 128 characters; at least ten distinct values |
| `order_date` | date/time | Valid, no later than `observation_end` |
| `gross_value` | numeric | Finite and non-negative |
| `contribution_margin` | numeric | Finite; may be negative |
| `discount_pct` | numeric | Between zero and one |
| `delivery_delay_days` | integer | Non-negative |
| `is_cancelled`, `is_returned` | integer | Zero or one; cannot both equal one |
| `category_id` | integer | Non-negative |

Identifiers are read as strings, so leading zeroes such as `001` are preserved. Values that
begin with `=`, `+`, `-`, or `@`, and values containing control characters, are rejected to
prevent spreadsheet-formula injection in downstream CSV handling.

## Resource and history limits

- 100 to 5,000,000 rows
- At most 250 MiB compressed input
- At least ten customers
- At least 180 calendar days between the first and last transaction

The defaults are explicit in `InputConfig`; deployments may instantiate stricter limits.

## Meaning of a retained purchase

A purchase is successful only when `is_cancelled = 0` and `is_returned = 0`. Returned orders
remain visible to experience/failure features but cannot anchor cadence, satisfy purchase-count
eligibility, or close a personalized purchase window.

## Provenance and privacy

The pipeline records the compressed-file SHA-256, byte size, row/customer counts, covered date
range, observation cutoff, and contract version. It deliberately omits the source path and
filename. Customer-facing CSV outputs contain a stable 24-character HMAC `customer_key`, never
the raw `customer_id`. Generated working datasets and fitted bundles belong in a governed output
directory and are ignored by Git.

## Claim boundary

Historical labels use purchases observed after each snapshot. Candidate selection, probability
calibration, and final testing use separate chronological periods. The `score` command creates
current operational rows without a future label. Neither path estimates incremental campaign
impact; that requires a randomized experiment or uplift design.
