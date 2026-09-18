from __future__ import annotations

import hashlib
import hmac
import os

import pandas as pd

SALT_ENVIRONMENT_VARIABLE = "CHURN_ID_SALT"
SYNTHETIC_SALT = "public-synthetic-personalized-churn-v2"


def resolve_salt(*, synthetic: bool) -> str:
    """Return a deterministic synthetic salt or require a deployment-owned secret."""

    if synthetic:
        return SYNTHETIC_SALT
    salt = os.environ.get(SALT_ENVIRONMENT_VARIABLE, "").strip()
    if len(salt) < 16:
        raise ValueError(
            f"{SALT_ENVIRONMENT_VARIABLE} must contain at least 16 characters for supplied data"
        )
    return salt


def pseudonymize_identifier(identifier: object, salt: str) -> str:
    digest = hmac.new(
        salt.encode("utf-8"), str(identifier).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return digest[:24]


def replace_customer_id(frame: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Return an export-safe copy with stable HMAC customer keys."""

    output = frame.copy()
    if "customer_id" in output:
        output.insert(
            output.columns.get_loc("customer_id"),
            "customer_key",
            output["customer_id"].map(lambda value: pseudonymize_identifier(value, salt)),
        )
        output = output.drop(columns="customer_id")
    return output
