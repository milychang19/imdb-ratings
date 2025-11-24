# data_prep_time_splits.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

DATA_DIR = Path("data")
RAW_MOVIES_PATH = DATA_DIR / "movie_metadata.csv"
SPLITS_DIR = DATA_DIR / "time_splits"
PROCESSED_DIR = DATA_DIR / "processed"
# Configuration for the provided dataset (columns you listed).
# Target: average user vote; date: release_date -> we extract year.
TARGET_COL = "vote_average"
DATE_COL = "release_date"  # original date column in dataset
YEAR_COL = "release_year"  # derived year column created during cleaning

# These dataset columns exist per your header; director/actor columns are not
# provided in the new dataset so we set those to None and skip actor/director
# specific features that rely on them.
DIRECTOR_COL = None
ACTOR1_COL = None
ACTOR2_COL = None
ACTOR3_COL = None

GENRES_COL = "genres"
BUDGET_COL = "budget"
GROSS_COL = "revenue"

# Numeric base cols present in the new dataset
NUMERIC_BASE_COLS = [
    "runtime",
    "vote_count",
    "revenue",
    "budget",
    "popularity",
]

# ID columns to carry through for error analysis / metadata
ID_COLS = [
    "id",
    "title",
    DATE_COL,
    GENRES_COL,
    "original_language",
    "original_title",
    "imdb_id",
]


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with missing target or date
    required_for_time = [c for c in [TARGET_COL, DATE_COL] if c in df.columns]
    if required_for_time:
        df = df.dropna(subset=required_for_time)

    # Parse release date into a year column (YEAR_COL)
    if DATE_COL in df.columns:
        # safe parse; invalid dates become NaT and were dropped above
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df[YEAR_COL] = df[DATE_COL].dt.year.astype("Int64")

    # No director/actor requirements in this dataset — those columns aren't present

    # Basic sanity for numeric columns
    for col in NUMERIC_BASE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # target numeric
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Log transforms (safe log1p for zeros)
    if BUDGET_COL in df.columns:
        df["log_budget"] = np.log1p(df[BUDGET_COL].clip(lower=0))
    if GROSS_COL in df.columns:
        df["log_gross"] = np.log1p(df[GROSS_COL].clip(lower=0))

    # Ratios
    if set([BUDGET_COL, GROSS_COL]).issubset(df.columns):
        denom = df[BUDGET_COL].replace(0, np.nan)
        df["gross_to_budget_ratio"] = (df[GROSS_COL] / denom).replace(
            [np.inf, -np.inf], np.nan
        )

    for col in ["num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "movie_facebook_likes"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def add_genre_dummies(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-hot encode genres on combined data so columns match, then split back."""
    combined = pd.concat({"train": train, "test": test}, names=["split", "row_id"])

    # Normalize genres representation. The new dataset sometimes stores genres
    # as JSON-like lists (e.g. "[{'id': 18, 'name':'Drama'}, ...]"). We try
    # to handle several formats and produce a pipe-separated string for
    # pd.get_dummies.
    def _normalize_genre_cell(val):
        if pd.isna(val):
            return ""
        s = str(val).strip()
        # JSON-like list of dicts
        if s.startswith("[") and "name" in s:
            try:
                import ast

                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    names = [d.get("name") or d.get("title") for d in parsed if isinstance(d, dict)]
                    return "|".join([n for n in names if n])
            except Exception:
                pass
        # common separators
        if "|" in s:
            return s
        if "," in s:
            return "|".join([p.strip() for p in s.split(",") if p.strip()])
        return s

    genres_col = GENRES_COL
    if genres_col not in combined.columns:
        genres_col = "genres"

    combined[genres_col] = combined[genres_col].apply(_normalize_genre_cell)
    genres_dummies = combined[genres_col].fillna("").str.get_dummies(sep="|")
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

    # Use detected gross column
    if GROSS_COL not in train.columns:
        # Nothing to do if we don't have gross
        return train, test

    global_gross_mean = train[GROSS_COL].mean()

    def add_for_entity(role_prefix: str, col_name: str) -> None:
        nonlocal train, test
        if col_name not in train.columns:
            return

        grp = train.groupby(col_name)
        movie_count = grp.size()
        avg_gross = grp[GROSS_COL].mean()

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
    add_for_entity("director", DIRECTOR_COL)
    # Main actor
    add_for_entity("actor1", ACTOR1_COL)
    # Supporting actors (optional, but cheap)
    add_for_entity("actor2", ACTOR2_COL)
    add_for_entity("actor3", ACTOR3_COL)

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

    # Target encodings (train-only stats on target column)
    for col in [DIRECTOR_COL, ACTOR1_COL, ACTOR2_COL, ACTOR3_COL]:
        if col in train.columns and col is not None:
            tr_enc, te_enc = target_encode_single(train, test, col, TARGET_COL, prefix=col)
            train[tr_enc.name] = tr_enc
            test[te_enc.name] = te_enc

    for col in ["language", "country", "content_rating", "color"]:
        if col in train.columns:
            tr_enc, te_enc = target_encode_single(train, test, col, TARGET_COL, prefix=col)
            train[tr_enc.name] = tr_enc
            test[te_enc.name] = te_enc

    # Instead of filling numeric NaNs with column means, drop any rows that
    # have missing values in the features or target. This makes the pipeline
    # robust when the new dataset has missingness that shouldn't be imputed.

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

    # Drop any rows with missing values in selected numeric features or target
    if feature_cols:
        drop_cols_train = [*feature_cols]
        if TARGET_COL in train.columns:
            drop_cols_train.append(TARGET_COL)
        train = train.dropna(subset=drop_cols_train)

        drop_cols_test = [*feature_cols]
        if TARGET_COL in test.columns:
            drop_cols_test.append(TARGET_COL)
        test = test.dropna(subset=drop_cols_test)

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

    if YEAR_COL not in df_raw.columns:
        raise RuntimeError(f"Year column not found in data (looked for {YEAR_COL})")
    years = sorted(df_raw[YEAR_COL].unique())
    all_splits_meta = []

    for cutoff in years:
        train_mask = df_raw[YEAR_COL] <= cutoff
        test_mask = df_raw[YEAR_COL].between(cutoff + 1, cutoff + gap_years)

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