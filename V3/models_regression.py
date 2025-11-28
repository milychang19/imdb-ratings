# models_regression.py

from __future__ import annotations

from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R^2 for regression models.

    Predictions are clamped to [0, 10] to match IMDb-like scales.
    """
    y_pred = np.clip(y_pred, 0.0, 10.0)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Ridge-regularized linear regression with standardization."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_polynomial_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    degree: int = 2,
) -> Pipeline:
    """Polynomial features + Ridge regression.

    Note: degree > 2 is likely to overfit badly on this dataset.
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Tuple[lgb.LGBMRegressor, Dict]:
    """Train a LightGBM regressor with early stopping on a validation set."""
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid), (X_train, y_train)],
        eval_names=["valid", "train"],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    eval_results = model.evals_result_
    return model, eval_results


def make_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
    train_sizes: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute learning curve for an estimator.

    Returns:
      train_sizes,
      mean train RMSE per size,
      mean validation RMSE per size.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_rmse = np.sqrt(-train_scores).mean(axis=1)
    valid_rmse = np.sqrt(-valid_scores).mean(axis=1)
    return sizes, train_rmse, valid_rmse
