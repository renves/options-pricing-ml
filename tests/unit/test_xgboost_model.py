"""Unit tests for XGBoost model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.tree_based.xgboost_model import XGBoostPricer


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.sum(X_train, axis=1) + np.random.randn(n_samples) * 0.1

    X_val = np.random.randn(30, n_features)
    y_val = np.sum(X_val, axis=1) + np.random.randn(30) * 0.1

    return X_train, y_train, X_val, y_val


@pytest.fixture
def feature_names() -> list[str]:
    """Sample feature names."""
    return ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]


class TestXGBoostPricerInit:
    """Tests for XGBoostPricer initialization."""

    def test_default_params(self) -> None:
        """Test that default params are set correctly."""
        model = XGBoostPricer()

        assert model.params["objective"] == "reg:squarederror"
        assert model.params["max_depth"] == 6
        assert model.params["learning_rate"] == 0.1
        assert model.params["random_state"] == 42

    def test_custom_params(self) -> None:
        """Test that custom params override defaults."""
        custom_params = {"max_depth": 10, "learning_rate": 0.05}
        model = XGBoostPricer(params=custom_params)

        assert model.params["max_depth"] == 10
        assert model.params["learning_rate"] == 0.05

    def test_default_experiment_name(self) -> None:
        """Test default experiment name."""
        model = XGBoostPricer()
        assert model.experiment_name == "xgboost-baseline"

    def test_custom_experiment_name(self) -> None:
        """Test custom experiment name."""
        model = XGBoostPricer(experiment_name="my-experiment")
        assert model.experiment_name == "my-experiment"

    def test_model_initially_none(self) -> None:
        """Test that model is None before training."""
        model = XGBoostPricer()
        assert model.model is None


class TestXGBoostPricerTrain:
    """Tests for XGBoostPricer training."""

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_train_returns_metrics(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that training returns metrics dictionary."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        metrics = model.train(X_train, y_train, X_val, y_val)

        assert "train_rmse" in metrics
        assert "val_rmse" in metrics
        assert "train_r2" in metrics
        assert "val_r2" in metrics

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_train_sets_model(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that training sets the model attribute."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)

        assert model.model is not None

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_train_logs_params(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that training logs parameters to MLflow."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)

        mock_mlflow.log_params.assert_called_once()

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_train_logs_metrics(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that training logs metrics to MLflow."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)

        mock_mlflow.log_metrics.assert_called()


class TestXGBoostPricerPredict:
    """Tests for XGBoostPricer prediction."""

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_predict_returns_array(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that predict returns numpy array."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_val)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_val)

    def test_predict_before_train_raises_error(self) -> None:
        """Test that predict before training raises ValueError."""
        model = XGBoostPricer()

        with pytest.raises(ValueError, match="not trained"):
            model.predict(np.array([[1, 2, 3]]))


class TestXGBoostPricerEvaluate:
    """Tests for XGBoostPricer evaluation."""

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_evaluate_returns_metrics(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that evaluate returns test metrics."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)

        X_test = np.random.randn(20, 5)
        y_test = np.sum(X_test, axis=1)

        metrics = model.evaluate(X_test, y_test, log_to_mlflow=False)

        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert "test_mape" in metrics

    def test_evaluate_before_train_raises_error(self) -> None:
        """Test that evaluate before training raises ValueError."""
        model = XGBoostPricer()

        with pytest.raises(ValueError, match="not trained"):
            model.evaluate(np.array([[1]]), np.array([1]))


class TestXGBoostPricerSaveLoad:
    """Tests for XGBoostPricer save/load."""

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_save_and_load(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Test saving and loading model."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Train and save
        model = XGBoostPricer()
        model.train(X_train, y_train, X_val, y_val)

        model_path = tmp_path / "model.json"
        model.save(model_path)

        # Load and predict
        loaded_model = XGBoostPricer()
        loaded_model.load(model_path)

        predictions = loaded_model.predict(X_val)
        assert len(predictions) == len(X_val)

    def test_save_before_train_raises_error(self, tmp_path: Path) -> None:
        """Test that save before training raises ValueError."""
        model = XGBoostPricer()

        with pytest.raises(ValueError, match="not trained"):
            model.save(tmp_path / "model.json")

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        model = XGBoostPricer()

        with pytest.raises(FileNotFoundError):
            model.load(tmp_path / "nonexistent.json")


class TestXGBoostPricerMetrics:
    """Tests for metric calculations."""

    @patch("src.models.tree_based.xgboost_model.mlflow")
    def test_metrics_are_reasonable(
        self,
        mock_mlflow: MagicMock,
        sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that metrics are in reasonable ranges."""
        X_train, y_train, X_val, y_val = sample_data

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = XGBoostPricer()
        metrics = model.train(X_train, y_train, X_val, y_val)

        # RMSE should be positive
        assert metrics["train_rmse"] > 0
        assert metrics["val_rmse"] > 0

        # R² should be between -inf and 1
        assert metrics["train_r2"] <= 1.0
        assert metrics["val_r2"] <= 1.0

        # For this simple synthetic data, R² should be high
        assert metrics["train_r2"] > 0.5
