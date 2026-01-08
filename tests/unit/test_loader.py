"""Unit tests for data loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.loader import (
    get_feature_columns,
    load_and_prepare_data,
    load_options_data,
    prepare_features,
)


class TestLoadOptionsData:
    """Tests for load_options_data function."""

    def test_invalid_year_raises_error(self) -> None:
        """Test that invalid year raises ValueError."""
        with pytest.raises(ValueError, match="Invalid year"):
            load_options_data(year=1999)

        with pytest.raises(ValueError, match="Invalid year"):
            load_options_data(year=2101)

    def test_invalid_month_raises_error(self) -> None:
        """Test that invalid month raises ValueError."""
        with pytest.raises(ValueError, match="Invalid month"):
            load_options_data(year=2024, month=0)

        with pytest.raises(ValueError, match="Invalid month"):
            load_options_data(year=2024, month=13)

    @patch("src.data.loader.bq")
    def test_loads_data_with_month(self, mock_bq: MagicMock) -> None:
        """Test loading data for specific month."""
        mock_bq.get_options.return_value = pd.DataFrame({"a": [1, 2]})
        mock_bq.get_stocks.return_value = pd.DataFrame({"b": [3, 4]})

        options, stocks = load_options_data(year=2024, month=11)

        mock_bq.get_options.assert_called_once_with(year=2024, month=11)
        mock_bq.get_stocks.assert_called_once_with(year=2024, month=11)
        assert len(options) == 2
        assert len(stocks) == 2

    @patch("src.data.loader.bq")
    def test_loads_data_full_year(self, mock_bq: MagicMock) -> None:
        """Test loading data for full year."""
        mock_bq.get_options.return_value = pd.DataFrame({"a": [1]})
        mock_bq.get_stocks.return_value = pd.DataFrame({"b": [2]})

        options, stocks = load_options_data(year=2024, month=None)

        mock_bq.get_options.assert_called_once_with(year=2024)
        mock_bq.get_stocks.assert_called_once_with(year=2024)

    @patch("src.data.loader.bq")
    def test_creates_cache_dir(self, mock_bq: MagicMock, tmp_path: Path) -> None:
        """Test that cache directory is created."""
        mock_bq.get_options.return_value = pd.DataFrame()
        mock_bq.get_stocks.return_value = pd.DataFrame()

        cache_dir = tmp_path / "cache" / "nested"
        load_options_data(year=2024, month=1, cache_dir=cache_dir)

        assert cache_dir.exists()

    @patch("src.data.loader.bq")
    def test_download_failure_raises_runtime_error(
        self, mock_bq: MagicMock
    ) -> None:
        """Test that download failure raises RuntimeError."""
        mock_bq.get_options.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Failed to download"):
            load_options_data(year=2024, month=1)


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    @patch("src.data.loader.AdvancedFeatureEngineer")
    @patch("src.data.loader.OptionFeatureEngineer")
    def test_adds_core_features(
        self,
        mock_ofe: MagicMock,
        mock_afe: MagicMock,
    ) -> None:
        """Test that core features are added."""
        options = pd.DataFrame({
            "close": [1.0, 2.0],
            "implied_volatility": [0.3, 0.4],
        })
        stocks = pd.DataFrame({"close": [100.0, 101.0]})

        mock_fe_instance = MagicMock()
        mock_fe_instance.add_all_features.return_value = options.copy()
        mock_ofe.return_value = mock_fe_instance

        mock_afe_instance = MagicMock()
        mock_afe_instance.add_all_advanced_features.return_value = options.copy()
        mock_afe.return_value = mock_afe_instance

        result = prepare_features(options, stocks, include_advanced=True)

        mock_fe_instance.add_all_features.assert_called_once()
        mock_afe_instance.add_all_advanced_features.assert_called_once()
        assert len(result) == 2

    @patch("src.data.loader.OptionFeatureEngineer")
    def test_skips_advanced_features_when_disabled(
        self,
        mock_ofe: MagicMock,
    ) -> None:
        """Test that advanced features are skipped when disabled."""
        options = pd.DataFrame({
            "close": [1.0],
            "implied_volatility": [0.3],
        })
        stocks = pd.DataFrame({"close": [100.0]})

        mock_fe_instance = MagicMock()
        mock_fe_instance.add_all_features.return_value = options.copy()
        mock_ofe.return_value = mock_fe_instance

        with patch("src.data.loader.AdvancedFeatureEngineer") as mock_afe:
            prepare_features(options, stocks, include_advanced=False)
            mock_afe.assert_not_called()

    @patch("src.data.loader.OptionFeatureEngineer")
    def test_drops_rows_with_missing_iv(self, mock_ofe: MagicMock) -> None:
        """Test that rows with missing IV are dropped."""
        options = pd.DataFrame({
            "close": [1.0, 2.0, 3.0],
            "implied_volatility": [0.3, None, 0.5],
        })
        stocks = pd.DataFrame()

        mock_fe_instance = MagicMock()
        mock_fe_instance.add_all_features.return_value = options.copy()
        mock_ofe.return_value = mock_fe_instance

        result = prepare_features(options, stocks, include_advanced=False)

        assert len(result) == 2


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_excludes_identifier_columns(self) -> None:
        """Test that identifier columns are excluded."""
        df = pd.DataFrame({
            "ticker": ["A"],
            "underlying": ["B"],
            "feature1": [1.0],
        })

        features = get_feature_columns(df)

        assert "ticker" not in features
        assert "underlying" not in features
        assert "feature1" in features

    def test_excludes_date_columns(self) -> None:
        """Test that date columns are excluded."""
        df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2024-01-01")],
            "maturity_date": [pd.Timestamp("2024-12-01")],
            "feature1": [1.0],
        })

        features = get_feature_columns(df)

        assert "trade_date" not in features
        assert "maturity_date" not in features

    def test_excludes_target_columns(self) -> None:
        """Test that target columns are excluded."""
        df = pd.DataFrame({
            "close": [1.0],
            "implied_volatility": [0.3],
            "feature1": [1.0],
        })

        features = get_feature_columns(df)

        assert "close" not in features
        assert "implied_volatility" not in features

    def test_excludes_private_columns(self) -> None:
        """Test that private columns (starting with _) are excluded."""
        df = pd.DataFrame({
            "_private": [1],
            "_internal": [2],
            "feature1": [1.0],
        })

        features = get_feature_columns(df)

        assert "_private" not in features
        assert "_internal" not in features
        assert "feature1" in features

    def test_returns_sorted_list(self) -> None:
        """Test that feature list is sorted."""
        df = pd.DataFrame({
            "z_feature": [1],
            "a_feature": [2],
            "m_feature": [3],
        })

        features = get_feature_columns(df)

        assert features == ["a_feature", "m_feature", "z_feature"]


class TestLoadAndPrepareData:
    """Tests for load_and_prepare_data function."""

    @patch("src.data.loader.prepare_features")
    @patch("src.data.loader.load_options_data")
    def test_returns_correct_tuple(
        self,
        mock_load: MagicMock,
        mock_prepare: MagicMock,
    ) -> None:
        """Test that function returns correct tuple."""
        mock_load.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_prepare.return_value = pd.DataFrame({
            "implied_volatility": [0.3, 0.4],
            "feature1": [1.0, 2.0],
            "feature2": [3.0, 4.0],
        })

        df, features, target = load_and_prepare_data(
            year=2024, month=11, target="implied_volatility"
        )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(features, list)
        assert target == "implied_volatility"
        assert "feature1" in features
        assert "feature2" in features

    @patch("src.data.loader.prepare_features")
    @patch("src.data.loader.load_options_data")
    def test_raises_error_for_invalid_target(
        self,
        mock_load: MagicMock,
        mock_prepare: MagicMock,
    ) -> None:
        """Test that invalid target raises ValueError."""
        mock_load.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_prepare.return_value = pd.DataFrame({"col1": [1]})

        with pytest.raises(ValueError, match="Target column"):
            load_and_prepare_data(year=2024, target="nonexistent")

    @patch("src.data.loader.prepare_features")
    @patch("src.data.loader.load_options_data")
    def test_drops_rows_with_nan_target(
        self,
        mock_load: MagicMock,
        mock_prepare: MagicMock,
    ) -> None:
        """Test that rows with NaN target are dropped."""
        mock_load.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_prepare.return_value = pd.DataFrame({
            "implied_volatility": [0.3, None, 0.5],
            "feature1": [1.0, 2.0, 3.0],
        })

        df, _, _ = load_and_prepare_data(
            year=2024, target="implied_volatility"
        )

        assert len(df) == 2
