"""Time-series aware data splitting module.

CRITICAL: This module ensures NO data leakage by:
- Never shuffling data
- Always splitting chronologically
- Ensuring train dates < val dates < test dates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for train/val/test split results."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return (min_date, max_date) for training set."""
        return self.train["trade_date"].min(), self.train["trade_date"].max()

    @property
    def val_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return (min_date, max_date) for validation set."""
        return self.val["trade_date"].min(), self.val["trade_date"].max()

    @property
    def test_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return (min_date, max_date) for test set."""
        return self.test["trade_date"].min(), self.test["trade_date"].max()

    def validate_no_leakage(self) -> bool:
        """Verify no data leakage: train < val < test chronologically."""
        train_max = self.train["trade_date"].max()
        val_min = self.val["trade_date"].min()
        val_max = self.val["trade_date"].max()
        test_min = self.test["trade_date"].min()

        if train_max >= val_min:
            logger.error(f"LEAKAGE: train_max={train_max} >= val_min={val_min}")
            return False

        if val_max >= test_min:
            logger.error(f"LEAKAGE: val_max={val_max} >= test_min={test_min}")
            return False

        return True


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_column: str = "trade_date",
) -> SplitResult:
    """Split data chronologically into train/val/test sets.

    IMPORTANT: This function NEVER shuffles data. It maintains
    strict temporal ordering to prevent data leakage.

    Args:
        df: DataFrame with date column
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        date_column: Name of date column

    Returns:
        SplitResult with train, val, test DataFrames

    Raises:
        ValueError: If ratios don't sum to 1.0 or date column missing
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    # Sort by date - CRITICAL for preventing leakage
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    result = SplitResult(train=train, val=val, test=test)

    # Validate no leakage
    if not result.validate_no_leakage():
        raise RuntimeError("Data leakage detected in split!")

    logger.info(
        f"Split complete - Train: {len(train)} ({train_ratio:.0%}), "
        f"Val: {len(val)} ({val_ratio:.0%}), "
        f"Test: {len(test)} ({test_ratio:.0%})"
    )
    logger.info(f"Train period: {result.train_dates[0]} to {result.train_dates[1]}")
    logger.info(f"Val period: {result.val_dates[0]} to {result.val_dates[1]}")
    logger.info(f"Test period: {result.test_dates[0]} to {result.test_dates[1]}")

    return result


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    date_column: str = "trade_date",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward cross-validation splits.

    Walk-forward validation expands the training set over time while
    keeping the test set as a fixed forward window. This is the gold
    standard for time-series cross-validation.

    Example with n_splits=3:
        Split 1: Train [====    ] Test [==]
        Split 2: Train [======  ] Test [==]
        Split 3: Train [========] Test [==]

    Args:
        df: DataFrame with date column
        n_splits: Number of CV splits
        train_ratio: Initial training ratio (expands over splits)
        date_column: Name of date column

    Yields:
        Tuple of (train_df, test_df) for each split
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)

    # Calculate test size (fixed for all splits)
    test_size = int(n * (1 - train_ratio) / n_splits)

    for i in range(n_splits):
        # Training expands over time
        train_end = int(n * train_ratio) + (i * test_size)
        test_end = train_end + test_size

        if test_end > n:
            break

        train = df_sorted.iloc[:train_end].copy()
        test = df_sorted.iloc[train_end:test_end].copy()

        # Validate no overlap
        train_max = train[date_column].max()
        test_min = test[date_column].min()

        if train_max >= test_min:
            raise RuntimeError(f"Leakage in split {i}: {train_max} >= {test_min}")

        logger.info(
            f"Split {i + 1}/{n_splits}: "
            f"Train={len(train)} ({train[date_column].min()} to {train_max}), "
            f"Test={len(test)} ({test_min} to {test[date_column].max()})"
        )

        yield train, test


def get_X_y(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from DataFrame.

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column

    Returns:
        Tuple of (X, y) as numpy arrays
    """
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y
