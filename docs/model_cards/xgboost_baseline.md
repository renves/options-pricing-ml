# XGBoost Baseline Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Model Type** | XGBoost Gradient Boosted Trees |
| **Task** | Implied Volatility Prediction |
| **Framework** | XGBoost + scikit-learn |
| **Version** | 1.0.0 |
| **Last Updated** | 2026-01-08 |

## Intended Use

### Primary Use Cases
- Predict implied volatility (IV) for B3 stock options
- Serve as baseline for comparing advanced models (LSTM, VAE, ViT)
- Feature importance analysis for options pricing

### Out-of-Scope Uses
- Real-time trading decisions without human oversight
- Options markets outside B3 (Brazilian exchange)
- Exotic options (barrier, Asian, etc.)

## Training Data

### Source
- **Exchange**: B3 (Brazilian Stock Exchange)
- **Data Provider**: b3quant library v0.1.18
- **Period**: November 2024 (baseline)

### Dataset Size
- **Total Records**: ~50,000 option trades
- **Train Set**: 70% (chronological)
- **Validation Set**: 15% (chronological)
- **Test Set**: 15% (chronological)

### Features (60+)
Engineered using b3quant's OptionFeatureEngineer and AdvancedFeatureEngineer:

**Moneyness Features:**
- moneyness (S/K)
- log_moneyness
- is_itm, is_atm, is_otm flags

**Time Features:**
- days_to_maturity
- sqrt_time, inv_sqrt_time
- is_short_term, is_medium_term, is_long_term

**Volatility Features:**
- IV rank (10d, 30d, 60d)
- IV percentile
- Volatility skew
- Realized volatility
- Vol-of-vol

**Greeks Exposure:**
- Total gamma exposure
- Total vega exposure
- Delta-weighted volume

**Technical Indicators:**
- RSI (Relative Strength Index)
- Bollinger Bands (width, position)
- Momentum indicators

**Market Regime:**
- is_trending
- is_ranging
- is_volatile

## Training Procedure

### Hyperparameters (Default)
```python
{
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "random_state": 42
}
```

### Hyperparameter Tuning
- **Method**: Optuna (TPE Sampler)
- **Trials**: 50 (configurable)
- **Search Space**:
  - max_depth: [3, 10]
  - learning_rate: [0.01, 0.3] (log scale)
  - n_estimators: [50, 500]
  - subsample: [0.6, 1.0]
  - colsample_bytree: [0.6, 1.0]

### Data Preprocessing
- No shuffling (time-series data)
- Chronological train/val/test split
- NaN values removed for IV and close price

## Evaluation

### Metrics
| Metric | Validation | Test |
|--------|------------|------|
| RMSE | TBD | TBD |
| MAE | TBD | TBD |
| R² | TBD | TBD |
| MAPE | TBD | TBD |

*Note: Update after training*

### Evaluation Methodology
- **Split**: Time-series (NO shuffle)
- **Cross-Validation**: Walk-forward validation
- **Leakage Prevention**: Train dates < Val dates < Test dates

## Explainability

### SHAP Analysis
- TreeExplainer for feature importance
- Summary plots logged to MLflow
- Top 20 features tracked per run

### Top Features (Expected)
1. moneyness
2. days_to_maturity
3. iv_rank_30d
4. realized_volatility_30d
5. gamma_exposure

## Limitations

### Known Limitations
- **Market Conditions**: Trained on single month; may not generalize to extreme volatility regimes
- **Liquidity**: May underperform for illiquid options with wide bid-ask spreads
- **Corporate Events**: Does not account for earnings, dividends, or corporate actions
- **Time Decay**: Performance may degrade for very short-term options (<7 days)

### Failure Modes
- Extrapolation beyond training data range
- Sudden regime changes (market crashes)
- New option series without historical data

## Ethical Considerations

### Intended Audience
- Quantitative researchers
- Financial engineering students
- ML engineers building options pricing systems

### Risks
- **Financial Risk**: Model predictions should not be used as sole basis for trading decisions
- **Model Drift**: Performance will degrade over time; requires periodic retraining

### Mitigation
- Always use with human oversight
- Implement drift detection monitoring
- Regular backtesting on new data

## Technical Requirements

### Dependencies
- Python 3.10+
- xgboost >= 2.0.0
- mlflow >= 2.10.0
- shap >= 0.45.0
- b3quant >= 0.1.18

### Hardware
- **Training**: CPU sufficient (~5 min on modern laptop)
- **Inference**: CPU (<1ms per prediction)
- **Memory**: ~2GB RAM for training

## Usage

### Training
```bash
# Basic training
uv run python scripts/train_xgboost.py --year 2024 --month 11

# With hyperparameter tuning
uv run python scripts/train_xgboost.py --year 2024 --month 11 --tune --n-trials 50
```

### Loading from MLflow
```python
from src.models.tree_based.xgboost_model import XGBoostPricer

model = XGBoostPricer.load_from_mlflow(run_id="<run_id>")
predictions = model.predict(X_test)
```

### Inference
```python
from src.models.tree_based.xgboost_model import XGBoostPricer

model = XGBoostPricer()
model.load("models/xgboost_baseline.json")
predictions = model.predict(X_new)
```

## MLflow Tracking

### Experiment Name
`xgboost-baseline`

### Logged Artifacts
- Model (xgboost format)
- Feature importance (JSON)
- SHAP summary plot
- SHAP importance plot

### Tags
- `model_type`: tree
- `framework`: xgboost
- `status`: development/staging/production

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-08 | Initial baseline model |

## Contact

- **Repository**: https://github.com/renves/options-pricing-ml
- **Data Source**: https://github.com/renves/b3quant

## Citation

If using this model in research, please cite:
```
@software{options_pricing_ml,
  title = {Options Pricing ML: Machine Learning for Volatility Prediction},
  author = {Renan Alves},
  year = {2026},
  url = {https://github.com/renves/options-pricing-ml}
}
```
