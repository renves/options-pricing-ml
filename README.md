# Options Pricing ML

> End-to-end machine learning system for options pricing and volatility surface modeling using state-of-the-art deep learning techniques.

## 🎯 Project Overview

Production-grade ML platform demonstrating:
- Advanced feature engineering for derivatives
- Multiple model architectures (XGBoost → LSTM → VAE → Vision Transformers)
- Complete MLOps pipeline (MLflow, Airflow, FastAPI)
- Rigorous model evaluation and comparison
- Production deployment with monitoring

**Data Source**: Brazilian stock exchange (B3) via [b3quant](https://github.com/renves/b3quant) library

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   B3Quant   │─────▶│   Features   │─────▶│  ML Models  │
│  (Library)  │      │  Engineering │      │  Training   │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │ Feature Store│      │   MLflow    │
                     │  (Parquet)   │      │  Registry   │
                     └──────────────┘      └─────────────┘
                                                  │
                                                  ▼
                                           ┌─────────────┐
                                           │   FastAPI   │
                                           │   Serving   │
                                           └─────────────┘
```

---

## 🚀 Quick Start

### Using Lightning.ai (Recommended)

1. **Create a Lightning.ai Studio** at [lightning.ai](https://lightning.ai)
2. **Clone this repository** in the Studio terminal
3. **Run the setup script**:

```bash
# Clone and setup
git clone https://github.com/renves/options-pricing-ml.git
cd options-pricing-ml
chmod +x setup_lightning.sh && ./setup_lightning.sh

# Train XGBoost model
uv run python scripts/train_xgboost.py

# View experiments in MLflow UI (Open Ports tab, port 5000)
```

### Local Development

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv sync

# Setup MLflow tracking (local)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
                     --default-artifact-root ./mlruns \
                     --host 0.0.0.0 --port 5000
```

### Using Docker (Full MLOps Stack)

```bash
# Start MLOps stack (MLflow + Postgres + MinIO)
docker-compose up -d

# Install dependencies
uv sync

# View experiments
open http://localhost:5000  # MLflow UI
```

---

## 📊 Models Implemented

| Model | Type | Status | Accuracy | Training Time |
|-------|------|--------|----------|---------------|
| Black-Scholes | Analytical | ✅ Baseline | - | Instant |
| XGBoost | Tree-based | ⏳ In Progress | TBD | ~5 min |
| LightGBM | Tree-based | ⏳ Planned | TBD | ~3 min |
| LSTM-GRU | Deep Learning | ⏳ Planned | TBD | ~30 min |
| VAE | Generative | ⏳ Planned | TBD | ~1 hour |
| Vision Transformer | Attention | ⏳ Planned | TBD | ~2 hours |
| PINN | Physics-Informed | ⏳ Planned | TBD | ~3 hours |

---

## 📈 Key Features

### Feature Engineering
- **60+ engineered features** including:
  - Moneyness metrics (S/K, log-moneyness)
  - Greeks exposure (gamma, vega, delta-hedged value)
  - Volatility metrics (IV rank, percentile, skew, vol-of-vol)
  - Technical indicators (RSI, Bollinger Bands)
  - Market regime detection (trending, ranging, volatile)

### Model Evaluation
- Cross-validation with time-series splits
- Multiple metrics: RMSE, MAE, MAPE, Sharpe ratio
- Statistical significance testing
- SHAP values for explainability

### MLOps
- Experiment tracking (MLflow)
- Model versioning and registry
- Automated retraining pipelines (Airflow)
- REST API serving (FastAPI)
- Monitoring and drift detection

---

## 🛠️ Tech Stack

**Core ML**:
- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- XGBoost, LightGBM
- PyTorch, TensorFlow

**MLOps**:
- MLflow (tracking & registry)
- Apache Airflow (orchestration)
- FastAPI (serving)
- Docker & Docker Compose

**Data**:
- b3quant (data source)
- Parquet (storage)
- DVC (versioning)

---

## 📁 Project Structure

```
options-pricing-ml/
├── data/                      # Data storage (gitignored)
│   ├── raw/                   # Raw B3 data via b3quant
│   ├── processed/             # Feature-engineered datasets
│   └── feature_store/         # Versioned features
├── src/
│   ├── data/                  # Data loading and validation
│   ├── features/              # Feature engineering pipelines
│   ├── models/                # Model implementations
│   │   ├── baseline/          # Black-Scholes, Heston
│   │   ├── tree_based/        # XGBoost, LightGBM
│   │   ├── deep_learning/     # LSTM, VAE, ViT
│   │   └── ensemble/          # Stacking, blending
│   ├── evaluation/            # Metrics, backtesting
│   └── serving/               # FastAPI app
├── notebooks/                 # EDA and experiments
├── tests/                     # Unit and integration tests
├── scripts/                   # Training and utility scripts
├── dags/                      # Airflow DAGs
├── docker-compose.yml         # MLOps stack
├── pyproject.toml             # Dependencies
└── README.md
```

---

## 📚 Documentation

- [ML Project Setup Guide](docs/ML_PROJECT_SETUP.md)
- [MLOps Roadmap](docs/ROADMAP.md)
- [Feature Engineering Guide](docs/FEATURES.md)
- [Model Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

## 🎓 Learning Objectives

This project demonstrates:
1. **Advanced ML Engineering**: Feature engineering, hyperparameter tuning, ensemble methods
2. **Deep Learning**: LSTM, VAE, Vision Transformers, Physics-Informed NNs
3. **MLOps**: End-to-end pipeline from experimentation to production
4. **Financial ML**: Domain-specific modeling for derivatives pricing
5. **Software Engineering**: Clean code, testing, CI/CD, documentation

---

## 📊 Results

> Results will be updated as models are trained

### Model Comparison (Test Set)

| Model | RMSE | MAE | MAPE | Sharpe Ratio |
|-------|------|-----|------|--------------|
| Black-Scholes | TBD | TBD | TBD | - |
| XGBoost | TBD | TBD | TBD | TBD |
| LSTM-GRU | TBD | TBD | TBD | TBD |
| VAE | TBD | TBD | TBD | TBD |

---

## 🤝 Contributing

This is a portfolio project. Feedback and suggestions are welcome via issues!

---

## 📄 License

MIT License

---

## 🔗 Related Projects

- [b3quant](https://github.com/renves/b3quant) - Data fetching library for B3
- [b3quant on PyPI](https://pypi.org/project/b3quant/)

---

**Status**: 🚧 Active Development | Phase 3 (Tree-Based Models)

**Last Updated**: 2026-01-04
