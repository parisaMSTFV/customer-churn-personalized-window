# Transaction-history input contract

`churn-pipeline run --input-transactions` accepts UTF-8 `.csv` and `.csv.gz` files under contract `transaction-history-v1.0`.

| Field | Type | Validation |
|---|---|---|
| `order_id` | string | Populated and unique |
| `customer_id` | string | Populated; at least ten distinct customers |
| `order_date` | date/time | Valid; total history spans at least 180 days |
| `gross_value` | numeric | Finite and non-negative |
| `contribution_margin` | numeric | Finite; may be negative |
| `discount_pct` | numeric | Between zero and one |
| `delivery_delay_days` | integer | Non-negative |
| `is_cancelled`, `is_returned` | integer | Zero or one; cannot both equal one |
| `category_id` | integer | Non-negative |

At least 100 transaction rows are required. Extra columns are ignored. The pipeline records the source filename, compressed-file SHA-256, row and customer counts, date range, and contract version. The input file itself is not copied, but derived modeling and priority artifacts retain `customer_id`; choose a governed output directory.

## Evaluation boundary

Personalized-window labels are constructed from later observed purchases, then training, calibration, and test data are split chronologically. Metrics therefore describe predictive ranking on the supplied history. They do not show that a retention contact causes incremental purchases, and they are not transportable beyond the supplied population without further validation.
