"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Generator

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_options_data() -> pd.DataFrame:
    """Create sample options data for testing."""
    return pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-03"]),
        "ticker": ["PETRA123", "PETRA123", "PETRA123"],
        "underlying": ["PETR4", "PETR4", "PETR4"],
        "option_type": ["CALL", "CALL", "CALL"],
        "strike": [35.0, 35.0, 35.0],
        "maturity_date": pd.to_datetime(["2024-12-20", "2024-12-20", "2024-12-20"]),
        "close": [2.50, 2.60, 2.55],
        "volume": [1000, 1500, 1200],
        "open_interest": [5000, 5100, 5050],
    })


@pytest.fixture
def sample_stocks_data() -> pd.DataFrame:
    """Create sample stocks data for testing."""
    return pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-03"]),
        "ticker": ["PETR4", "PETR4", "PETR4"],
        "close": [36.50, 36.80, 36.60],
        "volume": [50000000, 55000000, 52000000],
        "high": [37.00, 37.20, 37.00],
        "low": [36.00, 36.50, 36.20],
    })


@pytest.fixture
def sample_features_data(
    sample_options_data: pd.DataFrame,
    sample_stocks_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create sample data with basic features for testing."""
    df = sample_options_data.copy()
    df["spot_price"] = [36.50, 36.80, 36.60]
    df["moneyness"] = df["spot_price"] / df["strike"]
    df["days_to_maturity"] = (df["maturity_date"] - df["trade_date"]).dt.days
    df["implied_volatility"] = [0.35, 0.36, 0.355]
    return df


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    yield data_dir


@pytest.fixture(autouse=True)
def reset_random_seed() -> None:
    """Reset random seed before each test for reproducibility."""
    import numpy as np
    np.random.seed(42)
