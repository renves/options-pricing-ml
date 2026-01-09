"""FastAPI serving module."""

from src.serving.api import app
from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    "app",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "PredictionRequest",
    "PredictionResponse",
]
