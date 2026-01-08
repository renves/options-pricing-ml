"""XGBoost model for options pricing with MLflow tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class XGBoostPricer:
    """XGBoost model for options pricing with MLflow tracking.

    This class wraps XGBoost with built-in MLflow experiment tracking,
    making it easy to log parameters, metrics, and model artifacts.

    Example:
        >>> model = XGBoostPricer()
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        experiment_name: str = "xgboost-baseline",
    ) -> None:
        """Initialize XGBoost pricer.

        Args:
            params: XGBoost hyperparameters. Uses defaults if None.
            experiment_name: MLflow experiment name for tracking.
        """
        self.params = params or self._default_params()
        self.experiment_name = experiment_name
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] | None = None

    @staticmethod
    def _default_params() -> dict[str, Any]:
        """Default XGBoost hyperparameters."""
        return {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Train model with MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: Optional list of feature names
            run_name: Name for MLflow run
            tags: Additional tags for MLflow run

        Returns:
            Dictionary of validation metrics
        """
        self.feature_names = feature_names

        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(self.params)

            # Add default tags
            default_tags = {
                "model_type": "tree",
                "framework": "xgboost",
                "status": "development",
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Train model
            logger.info(f"Training XGBoost with {len(X_train)} samples...")
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Calculate metrics
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)

            metrics = {
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "train_r2": float(r2_score(y_train, train_pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "val_mae": float(mean_absolute_error(y_val, val_pred)),
                "val_r2": float(r2_score(y_val, val_pred)),
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                input_example=X_train[:5] if len(X_train) >= 5 else X_train,
            )

            # Log feature importance
            if feature_names:
                self._log_feature_importance(feature_names)

            logger.info(f"Training complete. Run ID: {run.info.run_id}")
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.4f}")
            logger.info(f"Validation R²: {metrics['val_r2']:.4f}")

            return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        log_to_mlflow: bool = True,
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target
            log_to_mlflow: Whether to log metrics to MLflow

        Returns:
            Dictionary of test metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.predict(X_test)

        metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_r2": float(r2_score(y_test, predictions)),
            "test_mape": float(
                np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
            ),
        }

        if log_to_mlflow:
            mlflow.log_metrics(metrics)

        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Test MAPE: {metrics['test_mape']:.2f}%")

        return metrics

    def _log_feature_importance(self, feature_names: list[str]) -> None:
        """Log feature importance to MLflow."""
        if self.model is None:
            return

        importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))

        # Sort by importance
        sorted_importance = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Log top 20 features as metrics
        for i, (name, imp) in enumerate(sorted_importance[:20]):
            mlflow.log_metric(f"importance_{i + 1:02d}_{name[:30]}", float(imp))

        # Log full importance as artifact
        import json

        with open("feature_importance.json", "w") as f:
            json.dump(importance_dict, f, indent=2)
        mlflow.log_artifact("feature_importance.json")
        Path("feature_importance.json").unlink()

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Path | str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        logger.info(f"Model loaded from {path}")

    @classmethod
    def load_from_mlflow(cls, run_id: str) -> "XGBoostPricer":
        """Load model from MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            XGBoostPricer instance with loaded model
        """
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.xgboost.load_model(model_uri)

        instance = cls()
        instance.model = loaded_model
        logger.info(f"Model loaded from MLflow run: {run_id}")

        return instance
