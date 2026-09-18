from pathlib import Path

import pandas as pd
import pytest

from customer_churn.config import InputConfig, SimulationConfig
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
    actual, provenance = load_transactions(path, observation_end="2025-12-31")
    assert len(actual) == len(expected)
    assert provenance["contract_version"] == "transaction-history-v2.0"
    assert provenance["customers"] == 20
    assert len(str(provenance["sha256"])) == 64
    assert pd.api.types.is_datetime64_any_dtype(actual["order_date"])
    assert "input_file" not in provenance


def test_transaction_input_rejects_duplicate_order_ids(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    frame = _fixture(path)
    frame.loc[1, "order_id"] = frame.loc[0, "order_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(TransactionValidationError, match="order_id must be unique"):
        load_transactions(path, observation_end="2025-12-31")


def test_transaction_input_preserves_leading_zero_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    frame = _fixture(path)
    frame["customer_id"] = frame["customer_id"].str.replace("C", "00", regex=False)
    frame.to_csv(path, index=False)
    actual, _ = load_transactions(path, observation_end="2025-12-31")
    assert actual.iloc[0]["customer_id"].startswith("00")


def test_transaction_input_rejects_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    frame = _fixture(path)
    frame["unexpected"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(TransactionValidationError, match="exactly match"):
        load_transactions(path, observation_end="2025-12-31")


@pytest.mark.parametrize("unsafe", ["=1+1", "+SUM(A1)", "@handle", "-1"])
def test_transaction_input_rejects_spreadsheet_formula_identifiers(
    tmp_path: Path, unsafe: str
) -> None:
    path = tmp_path / "transactions.csv"
    frame = _fixture(path)
    frame.loc[0, "customer_id"] = unsafe
    frame.to_csv(path, index=False)
    with pytest.raises(TransactionValidationError, match="formula"):
        load_transactions(path, observation_end="2025-12-31")


def test_transaction_input_rejects_rows_after_observation_end(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    _fixture(path)
    with pytest.raises(TransactionValidationError, match="after observation_end"):
        load_transactions(path, observation_end="2024-01-01")


def test_transaction_input_enforces_file_limit(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    _fixture(path)
    with pytest.raises(TransactionValidationError, match="file-size"):
        load_transactions(
            path,
            observation_end="2025-12-31",
            limits=InputConfig(max_file_bytes=10),
        )


def test_transaction_input_rejects_wrong_extension(tmp_path: Path) -> None:
    path = tmp_path / "transactions.txt"
    path.write_text("not,csv", encoding="utf-8")
    with pytest.raises(TransactionValidationError, match=".csv"):
        load_transactions(path, observation_end="2025-12-31")
