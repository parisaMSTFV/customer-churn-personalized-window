from __future__ import annotations

import pandas as pd
import pytest

from customer_churn.privacy import (
    pseudonymize_identifier,
    replace_customer_id,
    resolve_salt,
)


def test_pseudonym_is_stable_and_salt_specific() -> None:
    first = pseudonymize_identifier("001", "a-long-enough-secret")
    assert first == pseudonymize_identifier("001", "a-long-enough-secret")
    assert first != pseudonymize_identifier("001", "another-long-secret")
    assert len(first) == 24


def test_export_replaces_raw_customer_id() -> None:
    frame = pd.DataFrame({"customer_id": ["001"], "score": [0.5]})
    output = replace_customer_id(frame, "a-long-enough-secret")
    assert "customer_id" not in output
    assert "customer_key" in output
    assert "001" not in output.to_csv(index=False)


def test_synthetic_salt_is_available_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHURN_ID_SALT", raising=False)
    assert len(resolve_salt(synthetic=True)) >= 16


def test_supplied_data_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHURN_ID_SALT", raising=False)
    with pytest.raises(ValueError, match="CHURN_ID_SALT"):
        resolve_salt(synthetic=False)


def test_supplied_data_accepts_long_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURN_ID_SALT", "0123456789abcdef")
    assert resolve_salt(synthetic=False) == "0123456789abcdef"
