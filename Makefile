.PHONY: install lock run external-smoke test lint check wheel-smoke clean

install:
	uv sync --frozen --all-extras

lock:
	uv lock

run:
	uv run churn-pipeline run

external-smoke:
	uv run churn-pipeline run --project-root /tmp/churn-source --customers 700 --seed 7
	CHURN_ID_SALT=local-supplied-smoke-secret uv run churn-pipeline run --project-root /tmp/churn-input --input-transactions /tmp/churn-source/data/generated/transactions.csv.gz --observation-end 2026-06-30
	CHURN_ID_SALT=local-scoring-smoke-secret uv run churn-pipeline score --input-transactions /tmp/churn-source/data/generated/transactions.csv.gz --observation-end 2026-06-30 --model-bundle /tmp/churn-source/models/personalized_churn_model.joblib --output-root /tmp/churn-score

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

check: lint test
	uv run python scripts/check_sensitive.py

wheel-smoke:
	uv build
	uv venv /tmp/churn-wheel-venv --clear
	uv pip install --python /tmp/churn-wheel-venv/bin/python dist/*.whl
	cd /tmp && /tmp/churn-wheel-venv/bin/churn-pipeline run --project-root /tmp/churn-wheel-run --customers 180 --seed 11
	cd /tmp && CHURN_ID_SALT=local-wheel-score-secret /tmp/churn-wheel-venv/bin/churn-pipeline score --input-transactions /tmp/churn-wheel-run/data/generated/transactions.csv.gz --observation-end 2026-06-30 --model-bundle /tmp/churn-wheel-run/models/personalized_churn_model.joblib --output-root /tmp/churn-wheel-score

clean:
	rm -rf data/generated models .pytest_cache .ruff_cache
