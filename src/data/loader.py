"""Data loading module using b3quant library."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from b3quant.features import AdvancedFeatureEngineer, OptionFeatureEngineer

logger = logging.getLogger(__name__)


def load_options_data(
    year: int,
    month: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load options and stocks data from B3 via b3quant.

    Args:
        year: Year to download (e.g., 2024)
        month: Optional month (1-12). If None, downloads full year.
        cache_dir: Directory for caching. Defaults to ./data/raw

    Returns:
        Tuple of (options_df, stocks_df)

    Raises:
        ValueError: If year or month is invalid
        RuntimeError: If data download fails
    """
    import b3quant as bq

    if year < 2000 or year > 2100:
        raise ValueError(f"Invalid year: {year}")
    if month is not None and (month < 1 or month > 12):
        raise ValueError(f"Invalid month: {month}")

    cache_dir = cache_dir or Path("./data/raw")
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data for {year}/{month or 'full year'}")

    try:
        if month:
            options = bq.get_options(year=year, month=month)
            stocks = bq.get_stocks(year=year, month=month)
        else:
            options = bq.get_options(year=year)
            stocks = bq.get_stocks(year=year)
    except Exception as e:
        raise RuntimeError(f"Failed to download data: {e}") from e

    logger.info(f"Loaded {len(options)} options and {len(stocks)} stocks records")

    return options, stocks


def prepare_features(
    options: pd.DataFrame,
    stocks: pd.DataFrame,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """Add all features using b3quant feature engineers.

    Args:
        options: Options DataFrame from b3quant
        stocks: Stocks DataFrame from b3quant
        include_advanced: Whether to include advanced features (default True)

    Returns:
        DataFrame with 60+ features ready for ML
    """
    from b3quant.features import AdvancedFeatureEngineer, OptionFeatureEngineer

    logger.info("Adding core features...")
    fe = OptionFeatureEngineer()
    df = fe.add_all_features(options, stocks)

    if include_advanced:
        logger.info("Adding advanced features...")
        afe = AdvancedFeatureEngineer()
        df = afe.add_all_advanced_features(df, stocks)

    # Drop rows with NaN in critical columns
    initial_rows = len(df)
    df = df.dropna(subset=["implied_volatility", "close"])
    dropped = initial_rows - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing IV or close price")

    logger.info(f"Prepared {len(df)} records with {len(df.columns)} features")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns for ML training.

    Excludes identifier columns, target columns, and date columns.

    Args:
        df: DataFrame with features

    Returns:
        List of feature column names
    """
    exclude_cols = {
        # Identifiers
        "ticker",
        "underlying",
        "option_type",
        # Dates
        "trade_date",
        "maturity_date",
        # Target variables (what we're predicting)
        "close",
        "implied_volatility",
        # Raw price data (use features instead)
        "open",
        "high",
        "low",
        "volume",
        "open_interest",
        "num_trades",
        "strike",
        # Derived but not features
        "spot_price",
        "time_to_maturity",
    }

    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and not col.startswith("_")
    ]

    return sorted(feature_cols)


def load_and_prepare_data(
    year: int,
    month: int | None = None,
    target: str = "implied_volatility",
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """Complete data loading and preparation pipeline.

    Args:
        year: Year to download
        month: Optional month
        target: Target variable name (default: implied_volatility)
        cache_dir: Cache directory

    Returns:
        Tuple of (prepared_df, feature_columns, target_column)
    """
    options, stocks = load_options_data(year, month, cache_dir)
    df = prepare_features(options, stocks)
    feature_cols = get_feature_columns(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Remove rows where target is NaN
    df = df.dropna(subset=[target])

    logger.info(f"Final dataset: {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"Target: {target}")

    return df, feature_cols, target
