# data_prep_time_splits.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

DATA_DIR = Path("data")
RAW_MOVIES_PATH = DATA_DIR / "movie_metadata.csv"
SPLITS_DIR = DATA_DIR / "time_splits"
PROCESSED_DIR = DATA_DIR / "processed"

NUMERIC_BASE_COLS = [
    "duration",
    "director_facebook_likes",
    "actor_1_facebook_likes",
    "actor_2_facebook_likes",
    "actor_3_facebook_likes",
    "cast_total_facebook_likes",
    "budget",
    "gross",
]

ID_COLS = [
    "movie_title",
    "title_year",
    "genres",
    "director_name",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
]

TARGET_COL = "imdb_score"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with missing target or year
    df = df.dropna(subset=[TARGET_COL, "title_year"])
    df["title_year"] = df["title_year"].astype(int)

    # Keep only rows with at least a director and main actor
    df = df.dropna(subset=["director_name", "actor_1_name"])

    # Basic sanity for numeric columns
    for col in NUMERIC_BASE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Log transforms (safe log1p for zeros)
    if "budget" in df.columns:
        df["log_budget"] = np.log1p(df["budget"].clip(lower=0))
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"].clip(lower=0))

    # Ratios
    if set(["budget", "gross"]).issubset(df.columns):
        denom = df["budget"].replace(0, np.nan)
        df["gross_to_budget_ratio"] = (df["gross"] / denom).replace(
            [np.inf, -np.inf], np.nan
        )

    for col in ["num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "movie_facebook_likes"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def add_genre_dummies(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-hot encode genres on combined data so columns match, then split back."""
    combined = pd.concat(
        {"train": train, "test": test},
        names=["split", "row_id"]
    )
    genres_dummies = combined["genres"].fillna("").str.get_dummies(sep="|")
    if genres_dummies.shape[1] > 0:
        genres_dummies = genres_dummies.add_prefix("genre_")
        combined = pd.concat([combined, genres_dummies], axis=1)
    # Split back
    train_out = combined.xs("train")
    test_out = combined.xs("test")
    return train_out, test_out


def target_encode_single(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str,
    min_samples_leaf: int = 5,
    smoothing: float = 10.0,
    prefix: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Smoothed target encoding using train only, applied to train and test.

    Returns (enc_train, enc_test).
    """
    if prefix is None:
        prefix = col

    global_mean = train[target].mean()

    # Compute means and counts on training data
    agg = train.groupby(col)[target].agg(["mean", "count"])
    counts = agg["count"]
    means = agg["mean"]

    # Smoothing (similar to Kaggle formula)
    smoothing_factor = 1 / (1 + np.exp(-(counts - min_samples_leaf) / smoothing))
    encodings = global_mean * (1 - smoothing_factor) + means * smoothing_factor

    # Map to train & test
    train_enc = train[col].map(encodings).fillna(global_mean)
    test_enc = test[col].map(encodings).fillna(global_mean)

    train_enc.name = f"te_{prefix}"
    test_enc.name = f"te_{prefix}"
    return train_enc, test_enc

def add_history_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-entity historical aggregates (train-only stats) for directors/actors.

    - director_movie_count, director_avg_gross
    - actor1_movie_count, actor1_avg_gross
    - actor2_movie_count, actor2_avg_gross
    - actor3_movie_count, actor3_avg_gross
    """
    train = train.copy()
    test = test.copy()

    if "gross" not in train.columns:
        # Nothing to do if we don't have gross
        return train, test

    global_gross_mean = train["gross"].mean()

    def add_for_entity(role_prefix: str, col_name: str) -> None:
        nonlocal train, test
        if col_name not in train.columns:
            return

        grp = train.groupby(col_name)
        movie_count = grp.size()
        avg_gross = grp["gross"].mean()

        # Map to train
        train[f"{role_prefix}_movie_count"] = train[col_name].map(movie_count).fillna(0)
        train[f"{role_prefix}_avg_gross"] = (
            train[col_name].map(avg_gross).fillna(global_gross_mean)
        )

        # Map to test (using train stats only)
        test[f"{role_prefix}_movie_count"] = test[col_name].map(movie_count).fillna(0)
        test[f"{role_prefix}_avg_gross"] = (
            test[col_name].map(avg_gross).fillna(global_gross_mean)
        )

    # Directors
    add_for_entity("director", "director_name")
    # Main actor
    add_for_entity("actor1", "actor_1_name")
    # Supporting actors (optional, but cheap)
    add_for_entity("actor2", "actor_2_name")
    add_for_entity("actor3", "actor_3_name")

    return train, test

def build_features_for_split(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds numeric + genre + target-encoded features for a given split."""
    train = basic_clean(train_raw)
    test = basic_clean(test_raw)

    # Add simple numeric features
    train = add_simple_features(train)
    test = add_simple_features(test)

    # Genre dummies with aligned columns
    train, test = add_genre_dummies(train, test)

    # NEW: director/actor historical counts & avg gross (train-only stats)
    train, test = add_history_features(train, test)

    # Target encodings (train-only stats on imdb_score)
    for col in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name"]:
        if col in train.columns:
            tr_enc, te_enc = target_encode_single(train, test, col, TARGET_COL, prefix=col)
            train[tr_enc.name] = tr_enc
            test[te_enc.name] = te_enc

    for col in ["language", "country", "content_rating", "color"]:
        if col in train.columns:
            tr_enc, te_enc = target_encode_single(train, test, col, TARGET_COL, prefix=col)
            train[tr_enc.name] = tr_enc
            test[te_enc.name] = te_enc

    # Fill remaining numeric NaNs with column means (separately for train & test)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(train[col].mean())  # use train mean for test

    # Final feature list:
    #   - keep ID columns (for error analysis / metadata)
    #   - keep ONLY numeric feature columns
    id_cols_existing = [c for c in ID_COLS if c in train.columns]

    candidate_feature_cols = [
        c for c in train.columns
        if c not in id_cols_existing and c != TARGET_COL
    ]
    feature_cols = [
        c for c in candidate_feature_cols
        if pd.api.types.is_numeric_dtype(train[c])
    ]

    # leak_cols = ["gross", "log_gross", "gross_to_budget_ratio"]
    # feature_cols = [c for c in feature_cols if c not in leak_cols]

    train_out = train[id_cols_existing + feature_cols + [TARGET_COL]]
    test_out = test[id_cols_existing + feature_cols + [TARGET_COL]]

    return train_out, test_out

def generate_time_based_splits(
    movies_path: Path = RAW_MOVIES_PATH,
    gap_years: int = 2,
    min_train_size: int = 300,
    min_test_size: int = 80,
) -> None:
    """Create multiple train/test CSVs where test years are strictly after train years.

    For each cutoff year T, we create:
      - train: movies with title_year <= T
      - test:  movies with title_year in [T+1, T+gap_years]

    Only splits with enough train/test rows are saved.
    """
    ensure_dirs()

    df_raw = pd.read_csv(movies_path)
    df_raw = basic_clean(df_raw)

    years = sorted(df_raw["title_year"].unique())
    all_splits_meta = []

    for cutoff in years:
        train_mask = df_raw["title_year"] <= cutoff
        test_mask = df_raw["title_year"].between(cutoff + 1, cutoff + gap_years)

        if train_mask.sum() < min_train_size or test_mask.sum() < min_test_size:
            continue

        train_raw = df_raw.loc[train_mask].copy()
        test_raw = df_raw.loc[test_mask].copy()

        train_proc, test_proc = build_features_for_split(train_raw, test_raw)

        test_years_str = f"{cutoff + 1}_to_{cutoff + gap_years}"
        train_path = SPLITS_DIR / f"train_until_{cutoff}.csv"
        test_path = SPLITS_DIR / f"test_{test_years_str}.csv"

        train_proc.to_csv(train_path, index=False)
        test_proc.to_csv(test_path, index=False)

        all_splits_meta.append(
            {
                "cutoff_year": cutoff,
                "train_rows": len(train_proc),
                "test_rows": len(test_proc),
                "train_path": str(train_path),
                "test_path": str(test_path),
            }
        )

    meta_df = pd.DataFrame(all_splits_meta)
    meta_path = SPLITS_DIR / "time_splits_summary.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved {len(all_splits_meta)} time-based splits; metadata -> {meta_path}")


if __name__ == "__main__":
    generate_time_based_splits()