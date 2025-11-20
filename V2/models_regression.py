# models_regression.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import learning_curve

import lightgbm as lgb


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Clamp predictions to a plausible IMDb range
    y_pred = np.clip(y_pred, 0.0, 10.0)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_linear_regression(X_train, y_train):
    """Ridge-regularized linear regression to avoid coefficient blowup."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_polynomial_regression(X_train, y_train, degree: int = 2):
    """Polynomial features + Ridge regression.

    NOTE: degree>2 is very likely to overfit badly on this dataset.
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
    num_leaves: int = 64,
    learning_rate: float = 0.05,
    n_estimators: int = 2000,
) -> Tuple[lgb.LGBMRegressor, Dict]:
    """Train LightGBM with early stopping and return model + eval history."""
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=63,
        learning_rate=learning_rate,
        n_estimators=800,   # much smaller
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,       # quiets a lot of spam
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=["train", "valid"],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
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
    """Return train sizes, mean train scores, mean valid scores (using negative MSE)."""
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

    train_rmse = np.sqrt(-train_scores)
    valid_rmse = np.sqrt(-valid_scores)

    return sizes, train_rmse.mean(axis=1), valid_rmse.mean(axis=1)
