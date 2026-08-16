from pathlib import Path

import pandas as pd
import pytest

from customer_churn.config import SimulationConfig
from customer_churn.input_data import TransactionValidationError, load_transactions
from customer_churn.simulate import simulate_transactions


def _fixture(path: Path) -> pd.DataFrame:
    _, transactions = simulate_transactions(
        SimulationConfig(n_customers=20, start_date="2023-01-01", end_date="2025-12-31", seed=7)
    )
    transactions.to_csv(path, index=False)
    return transactions


def test_transaction_input_records_checksum_and_normalizes_types(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    expected = _fixture(path)
    actual, provenance = load_transactions(path)
    assert len(actual) == len(expected)
    assert provenance["contract_version"] == "transaction-history-v1.0"
    assert provenance["customers"] == 20
    assert len(str(provenance["sha256"])) == 64
    assert pd.api.types.is_datetime64_any_dtype(actual["order_date"])


def test_transaction_input_rejects_duplicate_order_ids(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    frame = _fixture(path)
    frame.loc[1, "order_id"] = frame.loc[0, "order_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(TransactionValidationError, match="order_id must be unique"):
        load_transactions(path)
