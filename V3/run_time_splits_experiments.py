from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    SPLITS_DIR,
    TARGET_COL,
    YEAR_COL,
    GENRES_COL,
    BUDGET_COL,
    GROSS_COL,
    RESULTS_DIR,
    MODELS_DIR,
)
from evaluation_plots import (
    make_error_analysis_table,
    plot_actual_vs_predicted_overlay,
    plot_group_residuals_by_feature,
    plot_lgbm_training_curves,
    plot_learning_curve,
    plot_residuals_vs_predicted,
    plot_shap_summary_and_dependence,
)
from models_regression import (
    evaluate_regression,
    make_learning_curve,
    train_lightgbm,
    train_linear_regression,
    train_polynomial_regression,
)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers for columns / splits ----------

META_COL_CANDIDATES = [
    "movie_title",
    "title",
    YEAR_COL,
    GENRES_COL,
    BUDGET_COL,
    GROSS_COL,
]


def load_splits(
    min_test_year: Optional[int] = None,
    max_test_year: Optional[int] = None,
) -> List[Tuple[Path, Path]]:
    """
    Return list of (train_path, test_path) for all time-based splits,
    optionally filtered by the *test-year* window.

    A split is kept if its test-year interval [start, end] intersects
    the requested [min_test_year, max_test_year].
    """
    meta_path = SPLITS_DIR / "time_splits_summary.csv"
    meta = pd.read_csv(meta_path)

    pairs: List[Tuple[Path, Path]] = []

    for _, row in meta.iterrows():
        train_path = Path(row["train_path"])
        test_path = Path(row["test_path"])

        # Parse years from "test_YYYY_to_ZZZZ.csv"
        test_name = Path(test_path).stem
        test_start_year: Optional[int] = None
        test_end_year: Optional[int] = None

        if test_name.startswith("test_") and "_to_" in test_name:
            after_prefix = test_name[len("test_") :]
            start_str, end_str = after_prefix.split("_to_")
            try:
                test_start_year = int(start_str)
                test_end_year = int(end_str)
            except ValueError:
                pass

        # Year-based filtering (by test years)
        if min_test_year is not None and test_start_year is not None:
            if test_end_year < min_test_year:
                continue

        if max_test_year is not None and test_start_year is not None:
            if test_start_year > max_test_year:
                continue

        pairs.append((train_path, test_path))

    return pairs


def select_id_and_feature_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Decide which columns are metadata vs. numeric features.

    - ID/meta columns are kept only for analysis.
    - Feature columns are numeric and exclude the target.
    """
    id_cols = [c for c in META_COL_CANDIDATES if c in df.columns]
    candidate_features = [
        c for c in df.columns
        if c not in id_cols and c != TARGET_COL
    ]
    feature_cols = [
        c for c in candidate_features
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    return id_cols, feature_cols


@dataclass
class SplitRunResult:
    metrics: List[Dict]
    y_true: np.ndarray
    preds: Dict[str, np.ndarray]
    meta: pd.DataFrame
    lgbm_for_shap: Optional[object] = None
    X_for_shap: Optional[pd.DataFrame] = None


# ---------- single-split training ----------


def run_single_split(
    split_idx: int,
    train_path: Path,
    test_path: Path,
    use_lightgbm: bool,
    is_first_split: bool,
) -> SplitRunResult:
    """Train models on one time-based split and return metrics/predictions."""
    print(f"\n=== Time split {split_idx} ===")
    print(f"  train: {train_path.name}")
    print(f"  test : {test_path.name}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_cols, feature_cols = select_id_and_feature_columns(train_df)

    X_train_full = train_df[feature_cols]
    y_train_full = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].to_numpy()

    meta_test = test_df[id_cols].copy()

    # Train/validation split for models that need it
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
    )

    all_metrics: List[Dict] = []
    preds: Dict[str, np.ndarray] = {}
    lgbm_for_shap = None
    X_for_shap = None

    # --- Linear regression (Ridge) ---
    lr_model = train_linear_regression(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    preds["linear"] = y_pred_lr
    metrics_lr = evaluate_regression(y_test, y_pred_lr)
    metrics_lr.update({"model": "linear", "split": split_idx})
    all_metrics.append(metrics_lr)

    joblib.dump(lr_model, MODELS_DIR / f"linear_split_{split_idx}.joblib")

    # --- Polynomial regression (degree=2, Ridge) ---
    poly_degree = 2
    poly_model = train_polynomial_regression(X_train, y_train, degree=poly_degree)
    y_pred_poly = poly_model.predict(X_test)
    preds[f"poly_deg_{poly_degree}"] = y_pred_poly
    metrics_poly = evaluate_regression(y_test, y_pred_poly)
    metrics_poly.update({"model": f"poly_deg_{poly_degree}", "split": split_idx})
    all_metrics.append(metrics_poly)

    joblib.dump(
        poly_model,
        MODELS_DIR / f"poly_deg_{poly_degree}_split_{split_idx}.joblib",
    )

    # --- LightGBM ---
    if use_lightgbm:
        lgbm_model, eval_results = train_lightgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
        )
        y_pred_lgbm = lgbm_model.predict(X_test)
        preds["lgbm"] = y_pred_lgbm
        metrics_lgbm = evaluate_regression(y_test, y_pred_lgbm)
        metrics_lgbm.update({"model": "lgbm", "split": split_idx})
        all_metrics.append(metrics_lgbm)

        joblib.dump(lgbm_model, MODELS_DIR / f"lgbm_split_{split_idx}.joblib")

        if is_first_split:
            # Training curves
            plot_lgbm_training_curves(eval_results)
            # Keep reference for SHAP
            lgbm_for_shap = lgbm_model
            X_for_shap = X_train_full.copy()
            joblib.dump(lgbm_for_shap, MODELS_DIR / "lgbm_for_shap.joblib")
            X_for_shap.to_parquet(MODELS_DIR / "X_for_shap.parquet")

    return SplitRunResult(
        metrics=all_metrics,
        y_true=y_test,
        preds={k: v for k, v in preds.items()},
        meta=meta_test,
        lgbm_for_shap=lgbm_for_shap,
        X_for_shap=X_for_shap,
    )


# ---------- global aggregation / reporting ----------


def summarize_and_plot_results(
    all_metrics: List[Dict],
    all_y_true: List[float],
    preds_accum: Dict[str, List[float]],
    meta_frames: List[pd.DataFrame],
    use_lightgbm: bool,
) -> None:
    """Save metrics, global plots, residual analysis, error tables."""
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "metrics_by_split.csv", index=False)

    metrics_summary = (
        metrics_df.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metrics_summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("\n=== Metrics summary ===")
    print(metrics_summary)

    y_true_all = np.array(all_y_true)
    preds_all = {name: np.array(vals) for name, vals in preds_accum.items()}

    # Overall actual vs predicted / residuals
    plot_actual_vs_predicted_overlay(y_true_all, preds_all)
    plot_residuals_vs_predicted(y_true_all, preds_all)

    meta_all = pd.concat(meta_frames, ignore_index=True)

    # Group residuals by budget, using the strongest model available
    if BUDGET_COL in meta_all.columns:
        if use_lightgbm and "lgbm" in preds_all:
            model_name_for_bins = "lgbm"
        else:
            # fall back to the first model in preds_all
            model_name_for_bins = next(iter(preds_all.keys()))

        y_pred_for_bins = preds_all[model_name_for_bins]

        plot_group_residuals_by_feature(
            y_true_all,
            y_pred_for_bins,
            feature=meta_all[BUDGET_COL],
            feature_name=f"{BUDGET_COL}_{model_name_for_bins}",
        )

    # Error analysis table
    best_model = "lgbm" if use_lightgbm and "lgbm" in preds_all else "linear"
    errors_df = make_error_analysis_table(
        y_true_all,
        preds_all,
        meta_all,
        best_model=best_model,
    )
    errors_df.to_csv(RESULTS_DIR / "worst_errors_detailed.csv", index=False)


def build_learning_curves(
    last_train_path: Path,
    use_lightgbm: bool,
) -> None:
    """Optionally build learning curves on the most recent training split."""
    print("\n=== Building learning curves ===")
    last_train_df = pd.read_csv(last_train_path)

    id_cols, feature_cols = select_id_and_feature_columns(last_train_df)
    X_last = last_train_df[feature_cols]
    y_last = last_train_df[TARGET_COL]

    # Linear model learning curve
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    lr_estimator = Pipeline(
        [("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))]
    )
    sizes, lr_train_rmse, lr_valid_rmse = make_learning_curve(
        lr_estimator,
        X_last,
        y_last,
    )
    plot_learning_curve(sizes, lr_train_rmse, lr_valid_rmse, model_name="linear")

    # LightGBM learning curve (small model just for shape)
    if use_lightgbm:
        import lightgbm as lgb

        lgbm_estimator = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        sizes, lgbm_train_rmse, lgbm_valid_rmse = make_learning_curve(
            lgbm_estimator,
            X_last,
            y_last,
        )
        plot_learning_curve(
            sizes,
            lgbm_train_rmse,
            lgbm_valid_rmse,
            model_name="lgbm",
        )


# ---------- main orchestration ----------


def run_all_splits(
    use_lightgbm: bool = True,
    max_splits: Optional[int] = None,        # None = no limit
    build_learning_curves_flag: bool = False,
    min_test_year: Optional[int] = None,     # filter by test-year start
    max_test_year: Optional[int] = None,     # filter by test-year end
) -> None:
    pairs = load_splits(
        min_test_year=min_test_year,
        max_test_year=max_test_year,
    )

    if not pairs:
        raise RuntimeError(
            "No time-based split CSVs found after year filtering. "
            "Check your min_test_year/max_test_year or regenerate splits."
        )

    print(f"Found {len(pairs)} time-based splits after year filtering")

    if max_splits is not None and len(pairs) > max_splits:
        pairs = pairs[-max_splits:]
        print(f"Using {len(pairs)} most recent splits after applying max_splits={max_splits}")
    else:
        print(f"Using all {len(pairs)} splits in the filtered range")

    all_metrics: List[Dict] = []
    all_y_true: List[float] = []
    preds_accum: Dict[str, List[float]] = {}
    meta_frames: List[pd.DataFrame] = []

    lgbm_for_shap = None
    X_for_shap = None

    for idx, (train_path, test_path) in enumerate(pairs, start=1):
        is_first_split = idx == 1
        result = run_single_split(
            split_idx=idx,
            train_path=train_path,
            test_path=test_path,
            use_lightgbm=use_lightgbm,
            is_first_split=is_first_split,
        )

        all_metrics.extend(result.metrics)
        all_y_true.extend(result.y_true.tolist())
        meta_frames.append(result.meta)

        # Accumulate predictions for each model
        if not preds_accum:
            # initialize keys on first iteration
            for name in result.preds.keys():
                preds_accum[name] = []

        for name, y_pred in result.preds.items():
            preds_accum[name].extend(y_pred.tolist())

        # Store a single LightGBM model + data frame for SHAP
        if use_lightgbm and result.lgbm_for_shap is not None and lgbm_for_shap is None:
            lgbm_for_shap = result.lgbm_for_shap
            X_for_shap = result.X_for_shap

    # Aggregate metrics and plots
    summarize_and_plot_results(
        all_metrics=all_metrics,
        all_y_true=all_y_true,
        preds_accum=preds_accum,
        meta_frames=meta_frames,
        use_lightgbm=use_lightgbm,
    )

    # Optional learning curves on the last (most recent) training split
    if build_learning_curves_flag:
        last_train_path, _ = pairs[-1]
        build_learning_curves(last_train_path, use_lightgbm=use_lightgbm)
    else:
        print("\nSkipping learning curves (build_learning_curves_flag=False)")

    # SHAP plots for LightGBM
    if use_lightgbm and lgbm_for_shap is not None and X_for_shap is not None:
        plot_shap_summary_and_dependence(lgbm_for_shap, X_for_shap)


if __name__ == "__main__":
    run_all_splits(
        use_lightgbm=True,
        max_splits=None,
        build_learning_curves_flag=False,
        min_test_year=1980,
        max_test_year=2024,
    )