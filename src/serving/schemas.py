"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "CALL"
    PUT = "PUT"


class PredictionRequest(BaseModel):
    """Request schema for option pricing prediction."""

    spot_price: float = Field(..., gt=0, description="Current stock price (S)")
    strike: float = Field(..., gt=0, description="Strike price (K)")
    days_to_maturity: int = Field(..., ge=1, le=365, description="Days until expiration")
    option_type: OptionType = Field(..., description="CALL or PUT")
    risk_free_rate: float = Field(default=0.1075, ge=0, le=1, description="Risk-free rate (default: SELIC)")

    # Optional features for better predictions
    moneyness: float | None = Field(default=None, description="S/K ratio (calculated if not provided)")
    realized_volatility: float | None = Field(default=None, ge=0, le=5, description="Historical volatility")
    volume: int | None = Field(default=None, ge=0, description="Trading volume")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "spot_price": 36.50,
                    "strike": 35.00,
                    "days_to_maturity": 30,
                    "option_type": "CALL",
                    "risk_free_rate": 0.1075,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for option pricing prediction."""

    implied_volatility: float = Field(..., description="Predicted implied volatility")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_version: str = Field(..., description="Model version used")

    # Additional info
    moneyness: float = Field(..., description="Calculated moneyness (S/K)")
    time_to_maturity: float = Field(..., description="Time to maturity in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "implied_volatility": 0.35,
                    "confidence": 0.85,
                    "model_version": "xgboost-v1.0.0",
                    "moneyness": 1.043,
                    "time_to_maturity": 0.082,
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    options: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Error details")
