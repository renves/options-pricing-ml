"""Model explainability using SHAP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap

if TYPE_CHECKING:
    import xgboost as xgb

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explainability for tree models.

    Provides feature importance analysis, dependency plots,
    and individual prediction explanations.
    """

    def __init__(
        self,
        model: xgb.XGBRegressor,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self._shap_values: np.ndarray | None = None
        self._X: np.ndarray | None = None

    def compute_shap_values(
        self,
        X: np.ndarray,
        sample_size: int | None = None,
    ) -> np.ndarray:
        """Compute SHAP values for dataset.

        Args:
            X: Feature matrix
            sample_size: Optional sample size for large datasets

        Returns:
            SHAP values array
        """
        if sample_size and len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]

        logger.info(f"Computing SHAP values for {len(X)} samples...")
        self._X = X
        self._shap_values = self.explainer.shap_values(X)
        logger.info("SHAP values computed successfully")

        return self._shap_values

    def get_feature_importance(self) -> dict[str, float]:
        """Get mean absolute SHAP values per feature.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._shap_values is None:
            raise ValueError("Call compute_shap_values() first")

        mean_shap = np.abs(self._shap_values).mean(axis=0)

        if self.feature_names:
            return dict(zip(self.feature_names, mean_shap))
        else:
            return {f"feature_{i}": v for i, v in enumerate(mean_shap)}

    def plot_summary(
        self,
        max_display: int = 20,
        save_path: Path | str | None = None,
        log_to_mlflow: bool = True,
    ) -> None:
        """Create SHAP summary plot.

        Args:
            max_display: Maximum features to display
            save_path: Optional path to save figure
            log_to_mlflow: Whether to log to MLflow
        """
        if self._shap_values is None or self._X is None:
            raise ValueError("Call compute_shap_values() first")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            logger.info("Summary plot logged to MLflow")

        plt.close()

    def plot_bar(
        self,
        max_display: int = 20,
        save_path: Path | str | None = None,
        log_to_mlflow: bool = True,
    ) -> None:
        """Create SHAP bar plot (feature importance).

        Args:
            max_display: Maximum features to display
            save_path: Optional path to save figure
            log_to_mlflow: Whether to log to MLflow
        """
        if self._shap_values is None or self._X is None:
            raise ValueError("Call compute_shap_values() first")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Bar plot saved to {save_path}")

        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "shap_importance.png")
            logger.info("Bar plot logged to MLflow")

        plt.close()

    def plot_dependence(
        self,
        feature: str | int,
        interaction_feature: str | int | None = "auto",
        save_path: Path | str | None = None,
        log_to_mlflow: bool = True,
    ) -> None:
        """Create SHAP dependence plot for a feature.

        Args:
            feature: Feature name or index
            interaction_feature: Feature for interaction coloring
            save_path: Optional path to save figure
            log_to_mlflow: Whether to log to MLflow
        """
        if self._shap_values is None or self._X is None:
            raise ValueError("Call compute_shap_values() first")

        # Convert feature name to index if needed
        if isinstance(feature, str) and self.feature_names:
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = feature

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()

        feature_name = (
            self.feature_names[feature_idx]
            if self.feature_names
            else f"feature_{feature_idx}"
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Dependence plot saved to {save_path}")

        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), f"shap_dependence_{feature_name}.png")
            logger.info(f"Dependence plot for {feature_name} logged to MLflow")

        plt.close()

    def explain_prediction(
        self,
        X_single: np.ndarray,
        save_path: Path | str | None = None,
        log_to_mlflow: bool = True,
    ) -> dict[str, float]:
        """Explain a single prediction.

        Args:
            X_single: Single sample (1D or 2D array)
            save_path: Optional path to save figure
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Dictionary of feature contributions
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_single)

        # Create waterfall plot
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X_single[0],
                feature_names=self.feature_names,
            ),
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), "shap_waterfall.png")

        plt.close()

        # Return contributions as dict
        if self.feature_names:
            return dict(zip(self.feature_names, shap_values[0]))
        else:
            return {f"feature_{i}": v for i, v in enumerate(shap_values[0])}


def explain_model(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    sample_size: int = 1000,
    log_to_mlflow: bool = True,
) -> SHAPExplainer:
    """Convenience function to explain model and log to MLflow.

    Args:
        model: Trained XGBoost model
        X: Feature matrix
        feature_names: List of feature names
        sample_size: Sample size for SHAP computation
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Configured SHAPExplainer instance
    """
    explainer = SHAPExplainer(model, feature_names)
    explainer.compute_shap_values(X, sample_size=sample_size)

    # Generate plots
    explainer.plot_summary(log_to_mlflow=log_to_mlflow)
    explainer.plot_bar(log_to_mlflow=log_to_mlflow)

    # Log feature importance
    importance = explainer.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    if log_to_mlflow:
        for i, (name, value) in enumerate(sorted_importance[:20]):
            mlflow.log_metric(f"shap_importance_{i + 1:02d}_{name[:25]}", value)

    logger.info("Model explanation complete")

    return explainer
