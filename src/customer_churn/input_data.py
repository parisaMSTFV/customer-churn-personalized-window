from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "gross_value",
    "contribution_margin",
    "discount_pct",
    "delivery_delay_days",
    "is_cancelled",
    "is_returned",
    "category_id",
]


class TransactionValidationError(ValueError):
    """Raised when supplied order history violates the input contract."""


def load_transactions(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate a CSV transaction history with content provenance."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes not in ([".csv"], [".csv", ".gz"]):
        raise TransactionValidationError("Input must be a .csv or .csv.gz file")
    try:
        content = path.read_bytes()
        compression = "gzip" if suffixes[-1] == ".gz" else None
        frame = pd.read_csv(
            io.BytesIO(content),
            compression=compression,
            dtype={"order_id": "string", "customer_id": "string"},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise TransactionValidationError(f"Could not read transaction input: {exc}") from exc

    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise TransactionValidationError(f"Missing required columns: {missing}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    numeric_columns = [
        "gross_value",
        "contribution_margin",
        "discount_pct",
        "delivery_delay_days",
        "is_cancelled",
        "is_returned",
        "category_id",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("order_id", "customer_id"):
        frame[column] = frame[column].str.strip()

    if len(frame) < 100:
        raise TransactionValidationError("At least 100 transaction rows are required")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise TransactionValidationError("Required transaction fields must not contain nulls")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise TransactionValidationError("Numeric transaction fields must be finite")
    if frame["order_id"].eq("").any() or frame["customer_id"].eq("").any():
        raise TransactionValidationError("Order and customer identifiers must be populated")
    if frame["order_id"].duplicated().any():
        raise TransactionValidationError("order_id must be unique")
    if frame["customer_id"].nunique() < 10:
        raise TransactionValidationError("At least ten customers are required")
    if (frame["order_date"].max() - frame["order_date"].min()) < pd.Timedelta(days=180):
        raise TransactionValidationError("Transaction history must span at least 180 days")
    if (frame["gross_value"] < 0).any():
        raise TransactionValidationError("gross_value must be non-negative")
    if not frame["discount_pct"].between(0, 1).all():
        raise TransactionValidationError("discount_pct must be between zero and one")
    for column in ("is_cancelled", "is_returned"):
        if not frame[column].isin([0, 1]).all():
            raise TransactionValidationError(f"{column} must contain only zero or one")
    if ((frame["is_cancelled"] == 1) & (frame["is_returned"] == 1)).any():
        raise TransactionValidationError("An order cannot be both cancelled and returned")
    for column in ("delivery_delay_days", "category_id"):
        if (frame[column] < 0).any() or not np.equal(frame[column], np.floor(frame[column])).all():
            raise TransactionValidationError(f"{column} must contain non-negative integers")

    frame[["delivery_delay_days", "is_cancelled", "is_returned", "category_id"]] = frame[
        ["delivery_delay_days", "is_cancelled", "is_returned", "category_id"]
    ].astype(int)
    frame = frame.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    provenance = {
        "contract_version": "transaction-history-v1.0",
        "input_file": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(frame),
        "customers": int(frame["customer_id"].nunique()),
        "start_date": frame["order_date"].min().date().isoformat(),
        "end_date": frame["order_date"].max().date().isoformat(),
    }
    return frame, provenance
