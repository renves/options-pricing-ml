#!/bin/bash
set -e

echo "Installing uv package manager..."
pip install uv

echo "Installing project dependencies..."
uv sync
uv sync --group dev

echo "Starting MLOps stack..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 30

echo "Testing b3quant import..."
uv run python -c "import b3quant; print(f'b3quant version: {b3quant.__version__}')"

echo "Setup complete! MLflow UI available at http://localhost:5000"
