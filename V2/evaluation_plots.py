# evaluation_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

import shap


PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_lgbm_training_curves(eval_results: Dict, filename: str = "lgbm_train_valid_rmse.png") -> None:
    """Plot LightGBM train/valid RMSE over boosting iterations."""
    train_rmse = eval_results["train"]["rmse"]
    valid_rmse = eval_results["valid"]["rmse"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse, label="train")
    plt.plot(valid_rmse, label="valid")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("LightGBM training vs validation RMSE")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_rmse: np.ndarray,
    valid_rmse: np.ndarray,
    model_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, marker="o", label="Train RMSE")
    plt.plot(train_sizes, valid_rmse, marker="s", label="CV RMSE")
    plt.xlabel("Fraction of training data")
    plt.ylabel("RMSE")
    plt.title(f"Learning curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / f"learning_curve_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_polynomial_complexity(
    degrees: List[int],
    valid_rmse: List[float],
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, valid_rmse, marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Validation RMSE")
    plt.title("Polynomial regression model complexity")
    plt.xticks(degrees)
    plt.tight_layout()
    out = PLOTS_DIR / "polynomial_model_complexity.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_actual_vs_predicted_overlay(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    split_ids: np.ndarray | None = None,
    alpha: float = 0.4,
) -> None:
    """
    Create:
      1) One overlay plot with all models on the same axes (0–10).
      2) One plot per model (0–10 axes).

    All values are clipped to [0, 10] for visualization.
    """
    # Clip actuals to [0, 10] for plotting
    y_true = np.clip(np.asarray(y_true), 0.0, 10.0)

    # Clip all predictions to [0, 10] for plotting
    preds_clipped: Dict[str, np.ndarray] = {
        name: np.clip(np.asarray(y_pred), 0.0, 10.0)
        for name, y_pred in preds.items()
    }

    min_val, max_val = 0.0, 10.0  # fixed IMDb scale

    # ---------- 1) Overlay plot: all models together ----------
    plt.figure(figsize=(8, 8))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Ideal")

    for name, y_pred in preds_clipped.items():
        plt.scatter(y_true, y_pred, alpha=alpha, s=15, label=name)

    plt.xlabel("Actual IMDb score")
    plt.ylabel("Predicted IMDb score")
    plt.title("Actual vs predicted IMDb scores (all models, all splits)")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "actual_vs_predicted_all_models.png"
    plt.savefig(out, dpi=150)
    plt.close()

    # ---------- 2) One plot per model ----------
    for name, y_pred in preds_clipped.items():
        plt.figure(figsize=(8, 8))
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Ideal")
        plt.scatter(y_true, y_pred, alpha=alpha, s=15)

        plt.xlabel("Actual IMDb score")
        plt.ylabel(f"Predicted IMDb score ({name})")
        plt.title(f"Actual vs predicted IMDb scores – {name}")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.tight_layout()
        out = PLOTS_DIR / f"actual_vs_predicted_{name}.png"
        plt.savefig(out, dpi=150)
        plt.close()



def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
) -> None:
    for name, y_pred in preds.items():
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 5))
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.scatter(y_pred, residuals, alpha=0.4)
        plt.xlabel("Predicted IMDb score")
        plt.ylabel("Residual (actual - predicted)")
        plt.title(f"Residuals vs predicted: {name}")
        plt.tight_layout()
        out = PLOTS_DIR / f"residuals_vs_predicted_{name}.png"
        plt.savefig(out, dpi=150)
        plt.close()


def plot_group_residuals_by_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature: pd.Series,
    feature_name: str,
    n_bins: int = 5,
) -> None:
    residuals = y_true - y_pred
    df = pd.DataFrame(
        {
            "feature": feature,
            "residual": residuals,
        }
    ).dropna()

    df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin")["residual"].agg(["mean", "count"]).reset_index()

    plt.figure(figsize=(9, 5))
    sns.barplot(x="bin", y="mean", data=grouped)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.ylabel("Mean residual")
    plt.xlabel(feature_name)
    plt.title(f"Mean residual by {feature_name} bin")
    plt.tight_layout()
    out = PLOTS_DIR / f"residuals_by_{feature_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_shap_summary_and_dependence(
    model,
    X: pd.DataFrame,
    top_feature: str | None = None,
    sample_size: int = 1000,
) -> None:
    """Create SHAP summary plot and one dependence plot for LightGBM."""
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary
    shap.summary_plot(shap_values, X_sample, show=False)
    out = PLOTS_DIR / "shap_summary_lightgbm.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    # Dependence plot on top feature (or the first one)
    if top_feature is None:
        top_feature = X_sample.columns[np.abs(shap_values).mean(axis=0).argmax()]

    shap.dependence_plot(
        top_feature,
        shap_values,
        X_sample,
        show=False,
    )
    out = PLOTS_DIR / f"shap_dependence_{top_feature}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_genre_frequency_from_df(
    df: pd.DataFrame,
    prefix: str = "genre_",
    filename: str = "genre_frequency.png",
) -> None:
    """
    Bar plot of how often each genre appears (multi-hot aware).

    Each genre_* column is summed across rows, so a movie with
    Action|Comedy contributes 1 to both Action and Comedy.
    """
    # pick all multi-hot genre columns
    genre_cols = [c for c in df.columns if c.startswith(prefix)]
    if not genre_cols:
        print("No genre_* columns found; skipping genre frequency plot.")
        return

    counts = df[genre_cols].sum(axis=0)
    counts = counts.sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=counts.index.str.replace(prefix, "", regex=False),
        y=counts.values,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of movie occurrences")
    plt.xlabel("Genre")
    plt.title("Frequency of genres (multi-hot counts)")
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()


def plot_num_genres_per_movie(
    df: pd.DataFrame,
    prefix: str = "genre_",
    filename: str = "num_genres_per_movie.png",
) -> None:
    """
    Histogram of how many genres each movie has (1,2,3,...).

    Uses the multi-hot genre_* columns for each row.
    """
    genre_cols = [c for c in df.columns if c.startswith(prefix)]
    if not genre_cols:
        print("No genre_* columns found; skipping num-genres-per-movie plot.")
        return

    num_genres = df[genre_cols].sum(axis=1)

    plt.figure(figsize=(7, 5))
    plt.hist(num_genres, bins=range(int(num_genres.max()) + 2), align="left", rwidth=0.8)
    plt.xlabel("Number of genres assigned to a movie")
    plt.ylabel("Number of movies")
    plt.title("Distribution of number of genres per movie")
    plt.xticks(range(int(num_genres.max()) + 1))
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()

def plot_imdb_histograms_by_genre(
    df: pd.DataFrame,
    target_col: str = "imdb_score",
    prefix: str = "genre_",
    max_genres: int = 9,
    filename: str = "imdb_hist_by_genre.png",
) -> None:
    """
    Plot IMDb score distributions for the most common genres.

    Creates a grid of histograms: one subplot per genre_* column, showing
    the distribution of imdb_score for movies that have that genre.
    Each title includes the mean IMDb score for that genre.
    """
    # Pick multi-hot genre columns
    genre_cols = [c for c in df.columns if c.startswith(prefix)]
    if not genre_cols:
        print("No genre_* columns found; skipping IMDb-by-genre plots.")
        return

    # How common each genre is (multi-hot counts)
    counts = df[genre_cols].sum(axis=0).sort_values(ascending=False)

    # Take the top N genres
    top_genres = counts.head(max_genres).index.tolist()
    if not top_genres:
        print("No genres to plot; skipping.")
        return

    # Figure layout (e.g. 3x3 grid for 9 genres)
    n_genres = len(top_genres)
    n_cols = 3
    n_rows = int(np.ceil(n_genres / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    bins = np.linspace(0, 10, 21)  # 0–10 in 0.5 steps

    for idx, col in enumerate(top_genres):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        scores = df.loc[df[col] == 1, target_col].dropna()
        if len(scores) == 0:
            ax.set_visible(False)
            continue

        mean_score = scores.mean()

        ax.hist(scores, bins=bins, density=True, alpha=0.7)
        ax.set_title(
            f"{col.replace(prefix, '')} (mean={mean_score:.2f})",
            fontsize=10,
        )
        ax.set_xlim(0, 10)
        ax.set_xlabel("IMDb score")
        ax.set_ylabel("Density")

    # Hide any unused subplots
    for idx in range(len(top_genres), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].set_visible(False)

    fig.suptitle("IMDb score distributions by genre (top genres)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()


def make_error_analysis_table(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    best_model: str,
    top_n: int = 20,
    out_csv: str = "worst_errors.csv",
) -> pd.DataFrame:
    """Return and save a table of the worst predictions based on a chosen model."""
    df = meta.copy().reset_index(drop=True)
    df["actual"] = y_true

    for name, y_pred in preds.items():
        df[f"pred_{name}"] = y_pred
        df[f"abs_error_{name}"] = np.abs(y_true - y_pred)

    df_sorted = df.sort_values(f"abs_error_{best_model}", ascending=False).head(top_n)
    out_path = PLOTS_DIR / out_csv
    df_sorted.to_csv(out_path, index=False)
    return df_sorted
