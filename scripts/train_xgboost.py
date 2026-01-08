#!/usr/bin/env python
"""Training script for XGBoost baseline model.

This script demonstrates a complete ML pipeline with:
- Data loading from b3quant
- Feature engineering
- Time-series train/val/test split
- Model training with MLflow tracking
- Hyperparameter tuning with Optuna
- SHAP explainability

Example:
    Basic training:
        $ uv run python scripts/train_xgboost.py --year 2024 --month 11

    With hyperparameter tuning:
        $ uv run python scripts/train_xgboost.py --year 2024 --month 11 --tune --n-trials 50

    Quick test run:
        $ uv run python scripts/train_xgboost.py --year 2024 --month 11 --quick
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import click
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--year", type=int, default=2024, help="Year to train on")
@click.option("--month", type=int, default=None, help="Month to train on (optional)")
@click.option(
    "--experiment",
    default="xgboost-baseline",
    help="MLflow experiment name",
)
@click.option(
    "--target",
    default="implied_volatility",
    help="Target variable to predict",
)
@click.option("--tune/--no-tune", default=False, help="Run hyperparameter tuning")
@click.option("--n-trials", type=int, default=50, help="Number of Optuna trials")
@click.option("--quick/--no-quick", default=False, help="Quick test run with small data")
def main(
    year: int,
    month: int | None,
    experiment: str,
    target: str,
    tune: bool,
    n_trials: int,
    quick: bool,
) -> None:
    """Train XGBoost baseline model for options pricing.

    This script loads B3 options data, engineers features using b3quant,
    trains an XGBoost model, and logs everything to MLflow.
    """
    logger.info("=" * 60)
    logger.info("XGBoost Training Pipeline")
    logger.info("=" * 60)

    # Import here to avoid slow startup for --help
    from src.data.loader import get_feature_columns, load_and_prepare_data
    from src.data.splitter import get_X_y, time_series_split
    from src.evaluation.explainer import explain_model
    from src.models.tree_based.xgboost_model import XGBoostPricer
    from src.training.tuner import optimize_xgboost, quick_tune

    # Step 1: Load and prepare data
    logger.info(f"Step 1: Loading data for {year}/{month or 'full year'}...")

    try:
        df, feature_cols, target_col = load_and_prepare_data(
            year=year,
            month=month,
            target=target,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} records with {len(feature_cols)} features")
    logger.info(f"Target: {target_col}")

    # For quick test, sample data
    if quick:
        sample_size = min(1000, len(df))
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Quick mode: sampled {sample_size} records")
        n_trials = min(n_trials, 10)

    # Step 2: Split data
    logger.info("Step 2: Splitting data (time-series aware)...")

    split_result = time_series_split(df)

    X_train, y_train = get_X_y(split_result.train, feature_cols, target_col)
    X_val, y_val = get_X_y(split_result.val, feature_cols, target_col)
    X_test, y_test = get_X_y(split_result.test, feature_cols, target_col)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Validate no leakage
    if not split_result.validate_no_leakage():
        logger.error("DATA LEAKAGE DETECTED! Aborting.")
        sys.exit(1)
    logger.info("No data leakage detected")

    # Step 3: Hyperparameter tuning (optional)
    if tune:
        logger.info(f"Step 3: Hyperparameter tuning ({n_trials} trials)...")
        if quick:
            best_params = quick_tune(X_train, y_train, X_val, y_val, n_trials=n_trials)
        else:
            best_params = optimize_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=n_trials,
                experiment_name=f"{experiment}-tuning",
            )
        logger.info(f"Best parameters: {best_params}")
    else:
        logger.info("Step 3: Skipping hyperparameter tuning (using defaults)")
        best_params = None

    # Step 4: Train model
    logger.info("Step 4: Training XGBoost model...")

    run_name = f"xgboost-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model = XGBoostPricer(params=best_params, experiment_name=experiment)

    train_metrics = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=feature_cols,
        run_name=run_name,
        tags={
            "year": str(year),
            "month": str(month) if month else "full",
            "tuned": str(tune),
        },
    )

    logger.info(f"Training complete - Val RMSE: {train_metrics['val_rmse']:.4f}")

    # Step 5: Evaluate on test set
    logger.info("Step 5: Evaluating on test set...")

    test_metrics = model.evaluate(X_test, y_test, log_to_mlflow=True)

    # Step 6: Generate SHAP explanations
    logger.info("Step 6: Generating SHAP explanations...")

    try:
        sample_size = min(500, len(X_test))
        explain_model(
            model.model,
            X_test,
            feature_names=feature_cols,
            sample_size=sample_size,
            log_to_mlflow=True,
        )
        logger.info("SHAP explanations logged to MLflow")
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Target: {target_col}")
    logger.info("")
    logger.info("Validation Metrics:")
    logger.info(f"  RMSE: {train_metrics['val_rmse']:.4f}")
    logger.info(f"  MAE:  {train_metrics['val_mae']:.4f}")
    logger.info(f"  R²:   {train_metrics['val_r2']:.4f}")
    logger.info("")
    logger.info("Test Metrics:")
    logger.info(f"  RMSE: {test_metrics['test_rmse']:.4f}")
    logger.info(f"  MAE:  {test_metrics['test_mae']:.4f}")
    logger.info(f"  R²:   {test_metrics['test_r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['test_mape']:.2f}%")
    logger.info("")
    logger.info("View results: mlflow ui (http://localhost:5000)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
