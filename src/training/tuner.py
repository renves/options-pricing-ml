"""Hyperparameter tuning using Optuna with MLflow integration."""

from __future__ import annotations

import logging
from typing import Any, Callable

import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
import xgboost as xgb

logger = logging.getLogger(__name__)


def get_xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define XGBoost hyperparameter search space.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters for XGBoost
    """
    return {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    experiment_name: str = "xgboost-tuning",
    study_name: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Find best XGBoost hyperparameters using Optuna.

    Integrates with MLflow for tracking all trials. Each trial is logged
    as a nested run under the main optimization run.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials (default 50)
        experiment_name: MLflow experiment name
        study_name: Optuna study name (default: auto-generated)
        timeout: Optional timeout in seconds

    Returns:
        Dictionary with best hyperparameters

    Example:
        >>> best_params = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=100)
        >>> model = XGBoostPricer(params=best_params)
    """
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = get_xgboost_search_space(trial)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        predictions = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))

        return rmse

    # Create MLflow callback for Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_rmse",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    with mlflow.start_run(run_name="hyperparameter-optimization") as run:
        mlflow.set_tags({
            "model_type": "tree",
            "framework": "xgboost",
            "optimization": "optuna",
            "status": "tuning",
        })

        mlflow.log_params({
            "n_trials": n_trials,
            "timeout": timeout,
            "train_size": len(X_train),
            "val_size": len(X_val),
        })

        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        # Log best results
        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_rmse", best_value)

        logger.info(f"Optimization complete. Best RMSE: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Run ID: {run.info.run_id}")

    # Add fixed params that weren't tuned
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    return best_params


def get_optimization_history(study: optuna.Study) -> dict[str, list[float]]:
    """Extract optimization history from Optuna study.

    Args:
        study: Completed Optuna study

    Returns:
        Dictionary with trial numbers and values
    """
    return {
        "trial_number": [t.number for t in study.trials],
        "value": [t.value for t in study.trials if t.value is not None],
        "best_value": [
            min(t.value for t in study.trials[: i + 1] if t.value is not None)
            for i in range(len(study.trials))
        ],
    }


def quick_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
) -> dict[str, Any]:
    """Quick hyperparameter tuning without MLflow logging.

    Useful for fast experimentation without full tracking.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of trials (default 20)

    Returns:
        Best hyperparameters
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = get_xgboost_search_space(trial)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        predictions = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, predictions)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    return best_params
