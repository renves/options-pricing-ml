# B3Quant ML - Machine Learning Project Setup

Este documento contém instruções completas para criar e configurar o projeto de Machine Learning que usa a biblioteca `b3quant`.

---

## 🎯 Objetivo do Projeto

Construir um **end-to-end ML system** para pricing e trading de opções da B3, demonstrando expertise em:
- Feature engineering avançado
- Modelos state-of-the-art (XGBoost, LSTM, VAE, ViT, PINN)
- MLOps completo (MLflow, Airflow, FastAPI)
- Production deployment

**Portfolio alvo**: Machine Learning Engineer

---

## 📁 Estrutura do Projeto

```
b3quant-ml/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── docker-compose.yml
│
├── data/                          # Dados (não versionado)
│   ├── raw/                       # COTAHIST files
│   ├── processed/                 # Features engineered
│   └── external/                  # Benchmarks (IBOV, etc)
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configurações centralizadas
│   │
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py             # Load data using b3quant
│   │   ├── preprocessor.py       # Data cleaning
│   │   └── feature_store.py      # Feature storage & versioning
│   │
│   ├── features/                 # Feature engineering (usa b3quant)
│   │   ├── __init__.py
│   │   └── builder.py            # Wrapper sobre b3quant features
│   │
│   ├── models/                   # ML Models
│   │   ├── __init__.py
│   │   ├── base.py               # Base model class
│   │   │
│   │   ├── baseline/             # Tree-based models
│   │   │   ├── __init__.py
│   │   │   ├── xgboost_model.py
│   │   │   └── lightgbm_model.py
│   │   │
│   │   ├── deep_learning/        # Neural networks
│   │   │   ├── __init__.py
│   │   │   ├── lstm_model.py
│   │   │   ├── gru_model.py
│   │   │   ├── vae_model.py      # Variational Autoencoder
│   │   │   ├── vit_model.py      # Vision Transformer
│   │   │   └── pinn_model.py     # Physics-Informed NN
│   │   │
│   │   └── ensemble/             # Ensemble methods
│   │       ├── __init__.py
│   │       └── stacking.py
│   │
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── tuner.py              # Hyperparameter tuning (Optuna)
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py            # Custom metrics for options
│   │   ├── backtester.py         # Backtesting framework
│   │   └── explainer.py          # SHAP, LIME, attention viz
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── io.py
│
├── mlops/                        # MLOps infrastructure
│   ├── mlflow/
│   │   ├── Dockerfile
│   │   └── mlflow.env
│   │
│   ├── airflow/
│   │   ├── dags/
│   │   │   ├── data_pipeline.py
│   │   │   ├── training_pipeline.py
│   │   │   └── inference_pipeline.py
│   │   ├── Dockerfile
│   │   └── airflow.env
│   │
│   └── api/
│       ├── main.py               # FastAPI app
│       ├── models.py             # Pydantic schemas
│       ├── endpoints/
│       │   ├── predict.py
│       │   └── health.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── experiments/                  # Experimentos trackados
│   ├── experiment_001_baseline/
│   ├── experiment_002_lstm/
│   └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── scripts/                      # Utility scripts
│   ├── download_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── compare_models.py
│
└── docs/
    ├── architecture.md
    ├── model_cards/              # Model documentation
    └── deployment.md
```

---

## 🔧 Configuração Inicial

### 1. Dependências (pyproject.toml)

```toml
[project]
name = "b3quant-ml"
version = "0.1.0"
description = "Machine Learning models for B3 options pricing"
requires-python = ">=3.10"

dependencies = [
    # Core
    "b3quant>=0.1.17",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",

    # ML - Tree models
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",

    # ML - Deep Learning
    "torch>=2.0.0",
    "tensorflow>=2.15.0",
    "transformers>=4.30.0",

    # Hyperparameter tuning
    "optuna>=3.5.0",
    "ray[tune]>=2.9.0",

    # Explainability
    "shap>=0.44.0",
    "lime>=0.2.0",

    # MLOps
    "mlflow>=2.10.0",
    "great-expectations>=0.18.0",

    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.18.0",

    # Utilities
    "pydantic>=2.5.0",
    "click>=8.1.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]

api = [
    "fastapi>=0.108.0",
    "uvicorn[standard]>=0.25.0",
    "pydantic-settings>=2.1.0",
]

airflow = [
    "apache-airflow>=2.8.0",
]

[project.scripts]
train = "src.scripts.train:main"
evaluate = "src.scripts.evaluate:main"
```

### 2. Docker Compose (MLOps Stack)

```yaml
version: '3.8'

services:
  # MLflow Tracking Server
  mlflow:
    build: ./mlops/mlflow
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - ARTIFACT_ROOT=s3://mlflow-artifacts
    depends_on:
      - postgres
      - minio
    volumes:
      - ./mlflow-data:/mlflow
    networks:
      - ml-network

  # PostgreSQL (MLflow backend)
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - ml-network

  # MinIO (S3-compatible artifact storage)
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data
    networks:
      - ml-network

  # Airflow (Optional - for production)
  # airflow-webserver:
  #   build: ./mlops/airflow
  #   ports:
  #     - "8080:8080"
  #   depends_on:
  #     - postgres
  #   environment:
  #     - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@postgres:5432/airflow
  #   networks:
  #     - ml-network

volumes:
  postgres-data:
  minio-data:

networks:
  ml-network:
    driver: bridge
```

### 3. Configuração (.env)

```bash
# Data
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed

# B3Quant
B3QUANT_CACHE_DIR=./data/raw
B3QUANT_USE_PARQUET=true

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=b3quant-options

# Models
MODEL_DIR=./models
CHECKPOINT_DIR=./checkpoints

# Training
SEED=42
DEVICE=cuda  # or cpu
NUM_WORKERS=4
```

---

## 🚀 Workflow de Desenvolvimento

### Fase 1: Setup & Data Exploration (Semana 1)

```bash
# 1. Clone e setup
git clone <repo-url> b3quant-ml
cd b3quant-ml
uv sync

# 2. Download data
uv run python scripts/download_data.py --year 2024

# 3. Start MLflow
docker-compose up -d mlflow postgres minio

# 4. Jupyter EDA
uv run jupyter notebook notebooks/01_eda.ipynb
```

**Deliverables**:
- Data downloaded (2020-2024)
- EDA notebook com insights
- Data quality report

### Fase 2: Feature Engineering & Baseline (Semana 2-3)

```python
# notebooks/02_feature_engineering.ipynb

from b3quant import get_options, get_stocks
from b3quant.features import OptionFeatureEngineer, AdvancedFeatureEngineer

# Load data
options = get_options(year=2024)
stocks = get_stocks(year=2024)

# Engineer features
fe = OptionFeatureEngineer()
afe = AdvancedFeatureEngineer()

options_ml = fe.add_all_features(options, stocks)
options_ml = afe.add_all_advanced_features(options_ml, stocks)

# Save to feature store
options_ml.to_parquet('data/processed/features_2024.parquet')
```

```python
# src/models/baseline/xgboost_model.py

import xgboost as xgb
import mlflow

with mlflow.start_run(run_name="xgboost_baseline"):
    # Train model
    model = xgb.XGBRegressor(...)
    model.fit(X_train, y_train)

    # Log to MLflow
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"rmse": rmse, "mae": mae})
    mlflow.xgboost.log_model(model, "model")
```

**Deliverables**:
- Feature engineering pipeline
- XGBoost baseline (RMSE, MAE)
- SHAP explainability

### Fase 3: Deep Learning Models (Semana 4-6)

**Week 4: LSTM/GRU**
```python
# src/models/deep_learning/lstm_model.py

import torch
import torch.nn as nn

class LSTMIVPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

**Week 5: VAE for Volatility Surface**
```python
# src/models/deep_learning/vae_model.py

class VolatilitySurfaceVAE(nn.Module):
    """Compress IV surface to latent space"""
    def __init__(self, surface_dim, latent_dim):
        # Encoder: surface -> latent
        # Decoder: latent -> surface
        pass
```

**Week 6: Vision Transformer**
```python
# src/models/deep_learning/vit_model.py

from transformers import ViTModel

class VolatilitySurfaceViT:
    """Treat IV surface as image for ViT"""
    pass
```

**Deliverables**:
- 3+ DL models treinados
- Model comparison dashboard
- Attention visualizations

### Fase 4: MLOps & Production (Semana 7-8)

**Airflow DAG**:
```python
# mlops/airflow/dags/training_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('model_training', schedule_interval='@weekly')

download_data = PythonOperator(task_id='download', ...)
engineer_features = PythonOperator(task_id='features', ...)
train_model = PythonOperator(task_id='train', ...)
evaluate_model = PythonOperator(task_id='evaluate', ...)

download_data >> engineer_features >> train_model >> evaluate_model
```

**FastAPI**:
```python
# mlops/api/main.py

from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/xgboost_iv/production")

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = prepare_features(request)
    prediction = model.predict(features)
    return {"iv": float(prediction[0])}
```

**Deliverables**:
- MLflow tracking completo
- Airflow DAGs funcionando
- FastAPI serving
- Docker deployment

---

## 📊 Experimentos & Tracking

### Estrutura de Experimento

```
experiments/
└── experiment_001_xgboost_baseline/
    ├── README.md                 # Descrição do experimento
    ├── config.yaml              # Hyperparameters
    ├── train.py                 # Script de treino
    ├── results/
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   └── shap_summary.png
    └── mlflow_run_id.txt
```

### MLflow Best Practices

```python
import mlflow

# Set experiment
mlflow.set_experiment("iv_prediction")

with mlflow.start_run(run_name="xgboost_v1"):
    # Log params
    mlflow.log_params({
        "max_depth": 10,
        "learning_rate": 0.01,
        "n_estimators": 100
    })

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("plots/feature_importance.png")

    # Log dataset
    mlflow.log_input(
        mlflow.data.from_pandas(X_train),
        context="training"
    )
```

---

## 🧪 Testing Strategy

```python
# tests/unit/test_models.py

def test_xgboost_training():
    model = XGBoostIVModel()
    model.fit(X_train, y_train)
    assert model.is_fitted

def test_prediction_shape():
    predictions = model.predict(X_test)
    assert predictions.shape == (len(X_test),)

def test_prediction_range():
    predictions = model.predict(X_test)
    assert (predictions > 0).all()  # IV must be positive
```

---

## 📝 Model Cards

Cada modelo deve ter documentação:

```markdown
# Model Card: XGBoost IV Predictor

## Model Details
- **Name**: XGBoost Implied Volatility Predictor v1.0
- **Type**: Gradient Boosted Trees
- **Task**: Regression (IV prediction)
- **Date**: 2025-01-03

## Intended Use
Predict implied volatility for Brazilian options (B3)

## Training Data
- Period: 2020-2024
- Samples: 1.2M option contracts
- Features: 45 (moneyness, Greeks, time series, regime)

## Performance
- Train RMSE: 0.032
- Val RMSE: 0.045
- Test RMSE: 0.048

## Limitations
- Only works for European options
- Requires underlying price data
- Performance degrades for DTE < 7 days

## Ethical Considerations
For educational/research purposes only.
```

---

## 🎯 Success Metrics

### Model Performance
- [ ] Baseline RMSE < 0.05
- [ ] LSTM beats baseline by >10%
- [ ] Ensemble RMSE < 0.04

### MLOps
- [ ] All experiments tracked in MLflow
- [ ] API latency < 100ms
- [ ] Monitoring dashboard deployed

### Portfolio
- [ ] 5+ models implemented
- [ ] Complete documentation
- [ ] Live demo available

---

## 📚 References

**Papers**:
1. [Deep Learning Option Pricing (2024)](https://arxiv.org/html/2509.05911v1)
2. [Vision Transformers for Volatility (2025)](https://arxiv.org/html/2511.03046)
3. [Physics-Informed Neural Networks](https://arxiv.org/html/2209.10771)

**Books**:
1. Machine Learning for Options Trading (2025)
2. Hands-On Machine Learning (Géron)

**MLOps**:
1. [MLflow Documentation](https://mlflow.org/)
2. [Airflow MLOps Guide](https://www.astronomer.io/docs/learn/airflow-mlops)

---

## 🚧 Next Steps

1. **Criar repositório**: `b3quant-ml`
2. **Setup inicial**: pyproject.toml, docker-compose.yml
3. **Download dados**: 2020-2024
4. **EDA notebook**: Análise exploratória
5. **Baseline model**: XGBoost
6. **MLflow tracking**: Experimentos

**Pronto para começar!** 🚀
