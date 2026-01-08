"""Unit tests for data splitter module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splitter import (
    SplitResult,
    get_X_y,
    time_series_split,
    walk_forward_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample DataFrame with dates for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "trade_date": dates,
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randn(100),
    })


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_date_properties(self, sample_df: pd.DataFrame) -> None:
        """Test date property methods."""
        result = SplitResult(
            train=sample_df.iloc[:70],
            val=sample_df.iloc[70:85],
            test=sample_df.iloc[85:],
        )

        train_min, train_max = result.train_dates
        assert train_min == pd.Timestamp("2024-01-01")
        assert train_max == pd.Timestamp("2024-03-11")

    def test_validate_no_leakage_passes(self, sample_df: pd.DataFrame) -> None:
        """Test that valid split passes leakage check."""
        result = SplitResult(
            train=sample_df.iloc[:70],
            val=sample_df.iloc[70:85],
            test=sample_df.iloc[85:],
        )

        assert result.validate_no_leakage() is True

    def test_validate_no_leakage_detects_overlap(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that overlapping dates are detected."""
        # Create overlap: train includes dates that are also in val
        result = SplitResult(
            train=sample_df.iloc[:75],  # Overlaps with val
            val=sample_df.iloc[70:85],
            test=sample_df.iloc[85:],
        )

        assert result.validate_no_leakage() is False


class TestTimeSeriesSplit:
    """Tests for time_series_split function."""

    def test_splits_with_default_ratios(self, sample_df: pd.DataFrame) -> None:
        """Test splitting with default 70/15/15 ratios."""
        result = time_series_split(sample_df)

        assert len(result.train) == 70
        assert len(result.val) == 15
        assert len(result.test) == 15

    def test_splits_with_custom_ratios(self, sample_df: pd.DataFrame) -> None:
        """Test splitting with custom ratios."""
        result = time_series_split(
            sample_df,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert len(result.train) == 80
        assert len(result.val) == 10
        assert len(result.test) == 10

    def test_ratios_must_sum_to_one(self, sample_df: pd.DataFrame) -> None:
        """Test that ratios must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            time_series_split(
                sample_df,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.1,  # Sum = 0.9
            )

    def test_missing_date_column_raises_error(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that missing date column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            time_series_split(sample_df, date_column="nonexistent")

    def test_no_shuffle_maintains_order(self, sample_df: pd.DataFrame) -> None:
        """Test that data is not shuffled - order is preserved."""
        result = time_series_split(sample_df)

        # Check chronological order
        assert (
            result.train["trade_date"].max() < result.val["trade_date"].min()
        )
        assert result.val["trade_date"].max() < result.test["trade_date"].min()

    def test_handles_unsorted_input(self) -> None:
        """Test that unsorted input is handled correctly."""
        # Create shuffled data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "trade_date": dates,
            "value": range(100),
        })
        df_shuffled = df.sample(frac=1, random_state=42)

        result = time_series_split(df_shuffled)

        # Should still be chronologically ordered after split
        assert result.validate_no_leakage()
        assert (
            result.train["trade_date"].max() < result.val["trade_date"].min()
        )

    def test_train_before_val_before_test(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Verify strict temporal ordering: train < val < test."""
        result = time_series_split(sample_df)

        train_max = result.train["trade_date"].max()
        val_min = result.val["trade_date"].min()
        val_max = result.val["trade_date"].max()
        test_min = result.test["trade_date"].min()

        assert train_max < val_min, "Train must end before val starts"
        assert val_max < test_min, "Val must end before test starts"


class TestWalkForwardSplit:
    """Tests for walk_forward_split function."""

    def test_generates_correct_number_of_splits(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that correct number of splits is generated."""
        splits = list(walk_forward_split(sample_df, n_splits=3))
        assert len(splits) == 3

    def test_training_set_expands(self, sample_df: pd.DataFrame) -> None:
        """Test that training set grows over splits."""
        splits = list(walk_forward_split(sample_df, n_splits=3))

        train_sizes = [len(train) for train, _ in splits]

        # Each split should have more training data
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_no_leakage_in_any_split(self, sample_df: pd.DataFrame) -> None:
        """Test that no split has data leakage."""
        for train, test in walk_forward_split(sample_df, n_splits=5):
            train_max = train["trade_date"].max()
            test_min = test["trade_date"].min()
            assert train_max < test_min, "Leakage detected!"

    def test_missing_date_column_raises_error(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that missing date column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            list(walk_forward_split(sample_df, date_column="nonexistent"))


class TestGetXy:
    """Tests for get_X_y function."""

    def test_extracts_correct_shapes(self) -> None:
        """Test that X and y have correct shapes."""
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "target": [7, 8, 9],
        })

        X, y = get_X_y(df, ["f1", "f2"], "target")

        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_returns_numpy_arrays(self) -> None:
        """Test that function returns numpy arrays."""
        df = pd.DataFrame({
            "f1": [1.0],
            "target": [2.0],
        })

        X, y = get_X_y(df, ["f1"], "target")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_preserves_order(self) -> None:
        """Test that data order is preserved."""
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "target": [7, 8, 9],
        })

        X, y = get_X_y(df, ["f1", "f2"], "target")

        np.testing.assert_array_equal(X[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(y, [7, 8, 9])
