.PHONY: install run external-smoke test lint check clean

install:
	python -m pip install -e ".[dev]"

run:
	python -m customer_churn.cli run

external-smoke:
	python -m customer_churn.cli run --project-root /tmp/churn-source --customers 700 --seed 7
	python -m customer_churn.cli run --project-root /tmp/churn-input --input-transactions /tmp/churn-source/data/generated/transactions.csv.gz

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

check: lint test
	python scripts/check_sensitive.py

clean:
	rm -rf data/generated models .pytest_cache .ruff_cache
