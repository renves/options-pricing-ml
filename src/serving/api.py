"""FastAPI application for model serving."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
MODEL: Any = None
MODEL_VERSION = "xgboost-v1.0.0"
API_VERSION = "1.0.0"


def load_model() -> Any:
    """Load the trained model.

    Tries to load from:
    1. MLflow run ID (if MLFLOW_RUN_ID env var is set)
    2. Local file path (if MODEL_PATH env var is set)
    3. Default bundled model

    Returns:
        Loaded model instance
    """
    import xgboost as xgb

    # Try MLflow first
    mlflow_run_id = os.getenv("MLFLOW_RUN_ID")
    if mlflow_run_id:
        try:
            import mlflow.xgboost

            logger.info(f"Loading model from MLflow run: {mlflow_run_id}")
            return mlflow.xgboost.load_model(f"runs:/{mlflow_run_id}/model")
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")

    # Try local path
    model_path = os.getenv("MODEL_PATH", "models/xgboost_baseline.json")
    if Path(model_path).exists():
        logger.info(f"Loading model from: {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    # Return None if no model found (will use mock predictions)
    logger.warning("No model found. API will return mock predictions.")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global MODEL
    logger.info("Starting API server...")
    MODEL = load_model()
    if MODEL:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Running without model - mock predictions enabled")
    yield
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Options Pricing ML API",
    description="""
    Machine Learning API for options pricing and implied volatility prediction.

    ## Features
    - Predict implied volatility for B3 stock options
    - Batch predictions for multiple options
    - Real-time inference with XGBoost model

    ## Model
    - **Type**: XGBoost Gradient Boosted Trees
    - **Target**: Implied Volatility (IV)
    - **Features**: 60+ engineered features from b3quant

    ## Usage
    Send a POST request to `/predict` with option parameters.
    """,
    version=API_VERSION,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Prepare feature vector from prediction request.

    Args:
        request: Prediction request with option parameters

    Returns:
        Feature array for model prediction
    """
    # Calculate derived features
    moneyness = request.moneyness or (request.spot_price / request.strike)
    log_moneyness = np.log(moneyness)
    time_to_maturity = request.days_to_maturity / 365.0
    sqrt_time = np.sqrt(time_to_maturity)

    # Feature flags
    is_call = 1.0 if request.option_type.value == "CALL" else 0.0
    is_itm = 1.0 if (is_call and moneyness > 1) or (not is_call and moneyness < 1) else 0.0
    is_atm = 1.0 if 0.95 <= moneyness <= 1.05 else 0.0
    is_otm = 1.0 if not is_itm and not is_atm else 0.0

    # Time categories
    is_short_term = 1.0 if request.days_to_maturity <= 30 else 0.0
    is_medium_term = 1.0 if 30 < request.days_to_maturity <= 90 else 0.0
    is_long_term = 1.0 if request.days_to_maturity > 90 else 0.0

    # Build feature vector
    # Note: Order must match training features
    features = np.array([
        moneyness,
        log_moneyness,
        time_to_maturity,
        sqrt_time,
        1 / sqrt_time if sqrt_time > 0 else 0,
        request.risk_free_rate,
        is_call,
        is_itm,
        is_atm,
        is_otm,
        is_short_term,
        is_medium_term,
        is_long_term,
        request.realized_volatility or 0.3,  # Default RV
        request.volume or 1000,  # Default volume
    ])

    return features.reshape(1, -1)


def mock_prediction(request: PredictionRequest) -> float:
    """Generate mock prediction when model is not available.

    Uses a simple approximation based on moneyness and time.
    """
    moneyness = request.spot_price / request.strike
    time_factor = np.sqrt(request.days_to_maturity / 365.0)

    # Simple IV approximation
    base_iv = 0.30
    moneyness_adj = abs(1 - moneyness) * 0.5
    time_adj = (1 - time_factor) * 0.1

    return base_iv + moneyness_adj + time_adj


@app.get("/", tags=["Info"])
async def root():
    """API root endpoint."""
    return {
        "name": "Options Pricing ML API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict implied volatility",
)
async def predict(request: PredictionRequest):
    """Predict implied volatility for a single option.

    ## Parameters
    - **spot_price**: Current stock price
    - **strike**: Option strike price
    - **days_to_maturity**: Days until expiration
    - **option_type**: CALL or PUT
    - **risk_free_rate**: Risk-free interest rate (default: SELIC)

    ## Returns
    - **implied_volatility**: Predicted IV
    - **confidence**: Model confidence (0-1)
    - **model_version**: Model version used
    """
    try:
        moneyness = request.spot_price / request.strike
        time_to_maturity = request.days_to_maturity / 365.0

        if MODEL is not None:
            features = prepare_features(request)
            prediction = float(MODEL.predict(features)[0])
            confidence = 0.85  # Could be calculated from model uncertainty
        else:
            prediction = mock_prediction(request)
            confidence = 0.5  # Lower confidence for mock

        # Ensure IV is in reasonable range
        prediction = max(0.05, min(2.0, prediction))

        return PredictionResponse(
            implied_volatility=round(prediction, 4),
            confidence=round(confidence, 2),
            model_version=MODEL_VERSION if MODEL else "mock-v1.0.0",
            moneyness=round(moneyness, 4),
            time_to_maturity=round(time_to_maturity, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Batch predict implied volatility",
)
async def predict_batch(request: BatchPredictionRequest):
    """Predict implied volatility for multiple options.

    Maximum 100 options per request.
    """
    predictions = []
    for option in request.options:
        result = await predict(option)
        predictions.append(result)

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "XGBoost",
        "target": "implied_volatility",
        "loaded": MODEL is not None,
        "features_count": 15,  # Simplified feature set for API
        "training_data": "B3 Options (November 2024)",
    }


# Entry point for uvicorn
def main():
    """Run the API server."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
