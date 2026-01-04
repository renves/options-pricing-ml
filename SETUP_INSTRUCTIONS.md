# Setup Instructions for options-pricing-ml

Quick guide to get the project running.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- uv (recommended) or pip

## Quick Start

### 1. Create Repository on GitHub

```bash
# Using GitHub CLI
gh repo create options-pricing-ml --public --description "ML platform for options pricing using state-of-the-art deep learning"

# Or create manually at https://github.com/new
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/options-pricing-ml.git
cd options-pricing-ml

# Copy all files from new_repo_files/ to the repository root
# (You have these files ready in the b3quant/new_repo_files/ directory)

# Create directory structure
mkdir -p data/{raw,processed,feature_store}
mkdir -p src/{data,features,models,evaluation,serving,training,utils}
mkdir -p src/models/{baseline,tree_based,deep_learning,ensemble}
mkdir -p tests/{unit,integration}
mkdir -p scripts
mkdir -p notebooks
mkdir -p dags
mkdir -p docs/model_cards
mkdir -p models checkpoints

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/models/baseline/__init__.py
touch src/models/tree_based/__init__.py
touch src/models/deep_learning/__init__.py
touch src/models/ensemble/__init__.py
touch src/evaluation/__init__.py
touch src/serving/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create .gitkeep for empty directories
touch data/.gitkeep
touch models/.gitkeep
touch checkpoints/.gitkeep
```

### 3. Install Dependencies

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -e ".[dev]"
```

### 4. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env if needed (defaults should work for local development)
```

### 5. Start MLOps Stack

```bash
# Start MLflow, PostgreSQL, and MinIO
docker-compose up -d

# Wait for services to be ready (~30 seconds)
# Check status
docker-compose ps

# You should see:
# - mlflow (port 5000)
# - postgres (port 5432)
# - minio (ports 9000, 9001)
```

### 6. Verify Setup

```bash
# Check MLflow UI
open http://localhost:5000

# Check MinIO Console
open http://localhost:9001
# Login: minioadmin / minioadmin

# Run a simple test
uv run pytest tests/ -v
```

### 7. First Steps

```bash
# Download sample data
uv run python -c "
import b3quant as bq
options = bq.get_options(year=2024, month=11)
print(f'Downloaded {len(options)} options')
"

# Start Jupyter for exploration
uv run jupyter notebook
```

## Project Files Checklist

Essential files you need in the repository:

- ✅ README.md - Project overview
- ✅ CLAUDE.md - AI assistant instructions (DO NOT COMMIT)
- ✅ pyproject.toml - Dependencies and config
- ✅ docker-compose.yml - MLOps stack
- ✅ .gitignore - Ignore patterns
- ✅ .env.example - Environment template
- ✅ LICENSE - MIT license
- ✅ Makefile - Common commands
- ✅ ROADMAP.md - Development roadmap (DO NOT COMMIT)
- ✅ ML_PROJECT_SETUP.md - Detailed setup guide

## Common Commands

```bash
# Install dependencies
make install

# Run tests
make test

# Format code
make format

# Type check
make typecheck

# Start MLOps stack
make docker-up

# Stop MLOps stack
make docker-down

# All-in-one setup
make dev-setup
```

## Troubleshooting

### Docker Issues

```bash
# Reset Docker volumes (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### MLflow Connection Issues

```bash
# Check if MLflow is running
curl http://localhost:5000/health

# Check logs
docker-compose logs mlflow

# Restart MLflow
docker-compose restart mlflow
```

### Python Dependencies

```bash
# Reinstall dependencies
rm -rf .venv uv.lock
uv sync
```

## Next Steps

1. Read [ML_PROJECT_SETUP.md](ML_PROJECT_SETUP.md) for detailed project structure
2. Read [ROADMAP.md](ROADMAP.md) for development plan
3. Start with Phase 3.1: Tree-Based Models (XGBoost)
4. Track all experiments in MLflow

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b exp/xgboost-baseline

# 2. Develop and test
# ... write code ...
make test
make lint

# 3. Commit
git add .
git commit -m "exp: implement XGBoost baseline model"

# 4. Push and create PR
git push origin exp/xgboost-baseline
```

## Resources

- [b3quant Documentation](https://github.com/renves/b3quant)
- [MLflow Documentation](https://mlflow.org/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Ready to start!** 🚀

Next: Implement your first model in `src/models/tree_based/xgboost_model.py`
