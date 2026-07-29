.PHONY: install run test lint check clean

install:
	python -m pip install -e ".[dev]"

run:
	python -m customer_churn.cli run

test:
	python -m pytest

lint:
	python -m ruff check .

check: lint test
	python scripts/check_sensitive.py

clean:
	rm -rf data/generated models .pytest_cache .ruff_cache
