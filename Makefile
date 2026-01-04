.PHONY: help install test lint format clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code with black and ruff"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make clean        - Clean cache and build files"
	@echo "  make docker-up    - Start MLOps stack (MLflow, Postgres, MinIO)"
	@echo "  make docker-down  - Stop MLOps stack"
	@echo "  make mlflow       - Open MLflow UI"

install:
	uv sync

test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

lint:
	uv run ruff check src/ tests/

format:
	uv run --with black black src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage build/ dist/

docker-up:
	docker-compose up -d
	@echo "MLflow UI: http://localhost:5000"
	@echo "MinIO Console: http://localhost:9001"

docker-down:
	docker-compose down

mlflow:
	@echo "Opening MLflow UI at http://localhost:5000"
	@python -m webbrowser http://localhost:5000

# Development shortcuts
dev-setup: install docker-up
	@echo "Development environment ready!"
	@echo "MLflow: http://localhost:5000"
	@echo "MinIO: http://localhost:9001"

check: lint typecheck test
	@echo "All checks passed!"
