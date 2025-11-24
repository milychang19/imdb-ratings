# run_time_splits_experiments.py

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from data_prep_time_splits import (
    SPLITS_DIR,
    TARGET_COL,
    YEAR_COL,
    GENRES_COL,
    DIRECTOR_COL,
    ACTOR1_COL,
    ACTOR2_COL,
    ACTOR3_COL,
    BUDGET_COL,
    GROSS_COL,
)
from models_regression import (
    train_linear_regression,
    train_polynomial_regression,
    train_lightgbm,
    evaluate_regression,
    make_learning_curve,
)
from evaluation_plots import (
    plot_lgbm_training_curves,
    plot_learning_curve,
    plot_actual_vs_predicted_overlay,
    plot_residuals_vs_predicted,
    plot_group_residuals_by_feature,
    plot_shap_summary_and_dependence,
    make_error_analysis_table,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits() -> List[Tuple[Path, Path]]:
    """Return list of (train_path, test_path) for all time-based splits."""
    meta_path = SPLITS_DIR / "time_splits_summary.csv"
    meta = pd.read_csv(meta_path)
    pairs = []
    for _, row in meta.iterrows():
        pairs.append((Path(row["train_path"]), Path(row["test_path"])))
    return pairs


def run_all_splits(
    use_lightgbm: bool = True,
    max_splits: int | None = 8,           # limit number of splits (None = use all)
    build_learning_curves: bool = False,  # set True only when you really need curves
) -> None:
    all_metrics_rows = []

    all_y_true: List[float] = []
    all_meta_rows = []
    preds_accum: Dict[str, List[float]] = {
        "linear": [],
        "poly_deg_2": [],
    }
    if use_lightgbm:
        preds_accum["lgbm"] = []

    pairs = load_splits()
    if not pairs:
        raise RuntimeError("No time-based split CSVs found. Run data_prep_time_splits.py first.")

    print(f"Found {len(pairs)} time-based splits")
    if max_splits is not None and len(pairs) > max_splits:
        pairs = pairs[-max_splits:]
    print(f"Using {len(pairs)} most recent splits for experiments")

    first_split = True
    lgbm_for_shap = None
    X_for_shap = None

    for split_idx, (train_path, test_path) in enumerate(pairs, start=1):
        print(f"\n=== Time split {split_idx}/{len(pairs)} ===")
        print(f"  train: {train_path.name}, test: {test_path.name}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Separate ID/meta columns from features — use detected names when available
        id_cols = [
            c for c in [
                "movie_title",
                YEAR_COL,
                GENRES_COL,
                DIRECTOR_COL,
                ACTOR1_COL,
                ACTOR2_COL,
                ACTOR3_COL,
                BUDGET_COL,
                GROSS_COL,
            ]
            if c in train_df.columns
        ]

        # Candidate features = everything except IDs and target
        candidate_feature_cols = [
            c for c in train_df.columns
            if c not in id_cols and c != TARGET_COL
        ]
        # Keep only numeric feature columns
        numeric_feature_cols = [
            c for c in candidate_feature_cols
            if pd.api.types.is_numeric_dtype(train_df[c])
        ]
        feature_cols = numeric_feature_cols

        X_train_full = train_df[feature_cols]
        y_train_full = train_df[TARGET_COL]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL]

        # Keep meta for error analysis (aligned with y_test)
        meta_test = test_df[id_cols]
        all_meta_rows.append(meta_test)

        # Split train into train/valid for poly + LightGBM
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )

        # --- Linear Regression (Ridge) ---
        lr_model = train_linear_regression(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        metrics_lr = evaluate_regression(y_test, y_pred_lr)
        metrics_lr.update({"model": "linear", "split": split_idx})
        all_metrics_rows.append(metrics_lr)

        # SAVE linear model
        joblib.dump(
            lr_model,
            MODELS_DIR / f"linear_split_{split_idx}.joblib"
        )

        # --- Polynomial Regression (degree 2, Ridge) ---
        poly_degree = 2
        poly_model = train_polynomial_regression(X_train, y_train, degree=poly_degree)
        y_pred_poly = poly_model.predict(X_test)
        metrics_poly = evaluate_regression(y_test, y_pred_poly)
        metrics_poly.update({"model": f"poly_deg_{poly_degree}", "split": split_idx})
        all_metrics_rows.append(metrics_poly)

        # SAVE polynomial degree-2 model
        joblib.dump(
            poly_model,
            MODELS_DIR / f"poly_deg_{poly_degree}_split_{split_idx}.joblib"
        )

        # --- LightGBM ---
        if use_lightgbm:
            lgbm_model, eval_results = train_lightgbm(
                X_train, y_train, X_valid, y_valid
            )
            y_pred_lgbm = lgbm_model.predict(X_test)
            metrics_lgbm = evaluate_regression(y_test, y_pred_lgbm)
            metrics_lgbm.update({"model": "lgbm", "split": split_idx})
            all_metrics_rows.append(metrics_lgbm)

            # SAVE LightGBM model for this split
            joblib.dump(
                lgbm_model,
                MODELS_DIR / f"lgbm_split_{split_idx}.joblib"
            )

            # Save training curves & SHAP baseline only once (first split)
            if first_split:
                plot_lgbm_training_curves(eval_results)
                lgbm_for_shap = lgbm_model
                X_for_shap = X_train_full.copy()
                joblib.dump(lgbm_for_shap, MODELS_DIR / "lgbm_for_shap.joblib")
                X_for_shap.to_parquet(MODELS_DIR / "X_for_shap.parquet")

        # Accumulate preds & true labels for global plots
        all_y_true.extend(y_test.tolist())
        preds_accum["linear"].extend(y_pred_lr.tolist())
        preds_accum["poly_deg_2"].extend(y_pred_poly.tolist())
        if use_lightgbm:
            preds_accum["lgbm"].extend(y_pred_lgbm.tolist())

        first_split = False

    # --- Aggregate metrics ---
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics_by_split.csv", index=False)

    metrics_summary = (
        metrics_df.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metrics_summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print(metrics_summary)

    # --- Global plots across all splits ---
    y_true_all = np.array(all_y_true)
    preds_all = {name: np.array(vals) for name, vals in preds_accum.items()}

    plot_actual_vs_predicted_overlay(y_true_all, preds_all)
    plot_residuals_vs_predicted(y_true_all, preds_all)

    # Group residuals by budget (using LightGBM if available else linear)
    meta_all = pd.concat(all_meta_rows, ignore_index=True)
    if BUDGET_COL in meta_all.columns:
        if use_lightgbm:
            y_pred_for_bins = preds_all["lgbm"]
            model_name_for_bins = "lgbm"
        else:
            y_pred_for_bins = preds_all["linear"]
            model_name_for_bins = "linear"

        plot_group_residuals_by_feature(
            y_true_all,
            y_pred_for_bins,
            feature=meta_all[BUDGET_COL],
            feature_name=f"{BUDGET_COL}_{model_name_for_bins}",
        )

    # Error analysis table
    errors_df = make_error_analysis_table(
        y_true_all,
        preds_all,
        meta_all,
        best_model="lgbm" if use_lightgbm else "linear",
    )
    errors_df.to_csv(RESULTS_DIR / "worst_errors_detailed.csv", index=False)

    # --- Learning curves (optional) ---
    if build_learning_curves:
        print("\n=== Building learning curves ===")

        last_train_path, _ = pairs[-1]
        last_train_df = pd.read_csv(last_train_path)

        id_cols = [
            c for c in [
                "movie_title",
                YEAR_COL,
                GENRES_COL,
                DIRECTOR_COL,
                ACTOR1_COL,
                ACTOR2_COL,
                ACTOR3_COL,
                BUDGET_COL,
                GROSS_COL,
            ]
            if c in last_train_df.columns
        ]

        candidate_feature_cols = [
            c for c in last_train_df.columns
            if c not in id_cols and c != TARGET_COL
        ]
        numeric_feature_cols = [
            c for c in candidate_feature_cols
            if pd.api.types.is_numeric_dtype(last_train_df[c])
        ]
        feature_cols = numeric_feature_cols

        X_last = last_train_df[feature_cols]
        y_last = last_train_df[TARGET_COL]

        # Linear learning curve (Ridge)
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        lr_estimator = Pipeline(
            [("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))]
        )
        sizes, lr_train_rmse, lr_valid_rmse = make_learning_curve(
            lr_estimator, X_last, y_last
        )
        plot_learning_curve(sizes, lr_train_rmse, lr_valid_rmse, model_name="linear")

        # LightGBM learning curve (using sklearn wrapper)
        if use_lightgbm:
            import lightgbm as lgb

            lgbm_estimator = lgb.LGBMRegressor(
                objective="regression",
                num_leaves=63,
                learning_rate=0.05,
                n_estimators=200,   # small-ish; this is just for curves
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            sizes, lgbm_train_rmse, lgbm_valid_rmse = make_learning_curve(
                lgbm_estimator, X_last, y_last
            )
            plot_learning_curve(
                sizes, lgbm_train_rmse, lgbm_valid_rmse, model_name="lgbm"
            )
    else:
        print("\nSkipping learning curves (build_learning_curves=False)")

    # SHAP plots for LightGBM (global summary + top feature dependence)
    if use_lightgbm and lgbm_for_shap is not None and X_for_shap is not None:
        plot_shap_summary_and_dependence(lgbm_for_shap, X_for_shap)


if __name__ == "__main__":
    # tweak here if you ever want all splits / learning curves
    # e.g. run_all_splits(max_splits=None, build_learning_curves=True)
    run_all_splits()
