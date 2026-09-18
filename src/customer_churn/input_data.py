from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from customer_churn.config import InputConfig

CONTRACT_VERSION = "transaction-history-v2.0"
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
_UNSAFE_IDENTIFIER = re.compile(r"^[=+\-@]|[\x00-\x1f\x7f]")


class TransactionValidationError(ValueError):
    """Raised when supplied order history violates the input contract."""


def _validate_identifiers(frame: pd.DataFrame) -> None:
    for column in ("order_id", "customer_id"):
        values = frame[column]
        if values.isna().any() or values.eq("").any():
            raise TransactionValidationError(f"{column} must be populated")
        if values.str.len().gt(128).any():
            raise TransactionValidationError(f"{column} must not exceed 128 characters")
        if values.str.match(_UNSAFE_IDENTIFIER).any():
            raise TransactionValidationError(
                f"{column} contains a control character or spreadsheet-formula prefix"
            )


def load_transactions(
    path: Path,
    *,
    observation_end: str | pd.Timestamp,
    limits: InputConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a bounded CSV transaction history under the strict v2 contract."""

    limits = limits or InputConfig()
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes not in ([".csv"], [".csv", ".gz"]):
        raise TransactionValidationError("Input must be a .csv or .csv.gz file")
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise TransactionValidationError(f"Could not inspect transaction input: {exc}") from exc
    if file_size > limits.max_file_bytes:
        raise TransactionValidationError("Transaction input exceeds the configured file-size limit")

    try:
        content = path.read_bytes()
        compression = "gzip" if suffixes[-1] == ".gz" else None
        frame = pd.read_csv(
            io.BytesIO(content),
            compression=compression,
            dtype={"order_id": "string", "customer_id": "string"},
            keep_default_na=True,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise TransactionValidationError(f"Could not read transaction input: {exc}") from exc

    if list(frame.columns) != REQUIRED_COLUMNS:
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        extra = sorted(set(frame.columns).difference(REQUIRED_COLUMNS))
        raise TransactionValidationError(
            f"Columns must exactly match the v2 contract; missing={missing}, extra={extra}"
        )
    if not limits.min_rows <= len(frame) <= limits.max_rows:
        raise TransactionValidationError(
            f"Transaction rows must be between {limits.min_rows:,} and {limits.max_rows:,}"
        )

    for column in ("order_id", "customer_id"):
        frame[column] = frame[column].str.strip()
    _validate_identifiers(frame)

    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce", utc=True)
    frame["order_date"] = frame["order_date"].dt.tz_convert(None).dt.normalize()
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

    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise TransactionValidationError("Required transaction fields must not contain nulls")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise TransactionValidationError("Numeric transaction fields must be finite")
    if frame["order_id"].duplicated().any():
        raise TransactionValidationError("order_id must be unique")
    if frame["customer_id"].nunique() < limits.min_customers:
        raise TransactionValidationError(
            f"At least {limits.min_customers} distinct customers are required"
        )
    if (frame["order_date"].max() - frame["order_date"].min()) < pd.Timedelta(
        days=limits.min_history_days
    ):
        raise TransactionValidationError(
            f"Transaction history must span at least {limits.min_history_days} days"
        )
    boundary = pd.Timestamp(observation_end)
    if boundary.tzinfo is not None:
        boundary = boundary.tz_convert(None)
    boundary = boundary.normalize()
    if frame["order_date"].max() > boundary:
        raise TransactionValidationError("Transactions must not occur after observation_end")
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
        "contract_version": CONTRACT_VERSION,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": file_size,
        "rows": len(frame),
        "customers": int(frame["customer_id"].nunique()),
        "start_date": frame["order_date"].min().date().isoformat(),
        "end_date": frame["order_date"].max().date().isoformat(),
        "observation_end": boundary.date().isoformat(),
    }
    return frame, provenance
