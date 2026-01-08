#!/bin/bash
# Lightning.ai Studio Setup Script
# Run this when you first open the studio

set -e

echo "=== Lightning.ai Studio Setup ==="

# Install uv package manager
echo "Installing uv..."
pip install uv

# Install project dependencies
echo "Installing dependencies..."
uv sync
uv sync --group dev

# Start MLflow (uses SQLite by default - no Docker needed!)
echo "Starting MLflow tracking server..."
mkdir -p mlruns
nohup uv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

echo "Waiting for MLflow to start..."
sleep 5

# Verify setup
echo "Verifying setup..."
uv run python -c "
import b3quant
import mlflow
import xgboost
import shap

print(f'b3quant: {b3quant.__version__}')
print(f'mlflow: {mlflow.__version__}')
print(f'xgboost: {xgboost.__version__}')
print(f'shap: {shap.__version__}')
print('All dependencies OK!')
"

echo ""
echo "=== Setup Complete! ==="
echo "MLflow UI: http://localhost:5000"
echo ""
echo "Next steps:"
echo "1. Open MLflow UI in the 'Open Ports' tab"
echo "2. Run: uv run python scripts/train_xgboost.py"
