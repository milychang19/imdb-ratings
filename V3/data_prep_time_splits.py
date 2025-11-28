# data_prep_time_splits.py

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Generator, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    RAW_MOVIES_PATH,
    SPLITS_DIR,
    PROCESSED_DIR,
    TARGET_COL,
    DATE_COL,
    YEAR_COL,
    GENRES_COL,
    BUDGET_COL,
    GROSS_COL,
    NUMERIC_BASE_COLS,
    ID_COLS,
    CATEGORICAL_FOR_TARGET_ENCODING,
)


# ---------- basic utilities ----------


def ensure_dirs() -> None:
    """Make sure all output directories exist."""
    for directory in (PROCESSED_DIR, SPLITS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_release_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df[YEAR_COL] = df[DATE_COL].dt.year.astype("Int64")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: drop missing target/date, parse year, coerce numeric."""
    df = df.copy()

    required = [c for c in (TARGET_COL, DATE_COL) if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    df = _parse_release_year(df)
    df = _coerce_numeric(df, NUMERIC_BASE_COLS)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return df


# ---------- simple numeric features ----------


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic log / ratio features that are cheap and generally useful."""
    df = df.copy()

    if BUDGET_COL in df.columns:
        df["log_budget"] = np.log1p(df[BUDGET_COL].clip(lower=0))

    if GROSS_COL in df.columns:
        df["log_gross"] = np.log1p(df[GROSS_COL].clip(lower=0))

    if BUDGET_COL in df.columns and GROSS_COL in df.columns:
        denom = df[BUDGET_COL].replace(0, np.nan)
        ratio = df[GROSS_COL] / denom
        df["gross_to_budget_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    extra_log_cols = [
        "num_voted_users",
        "num_critic_for_reviews",
        "num_user_for_reviews",
        "movie_facebook_likes",
    ]
    for col in extra_log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# ---------- genres encoding ----------


def _normalize_genre_cell(val) -> str:
    """Turn various genre representations into a 'A|B|C' string."""
    if pd.isna(val):
        return ""

    s = str(val).strip()

    # JSON-like list of dicts
    if s.startswith("[") and "name" in s:
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                names = [
                    d.get("name") or d.get("title")
                    for d in parsed
                    if isinstance(d, dict)
                ]
                names = [n for n in names if n]
                return "|".join(names)
        except Exception:
            # If parsing fails, fall through to generic handling
            pass

    # Simple separators
    if "|" in s:
        return s
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return "|".join(parts)

    return s


def add_genre_dummies(
    train: pd.DataFrame,
    test: pd.DataFrame,
    genres_col: str = GENRES_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-hot encode genres using the combined train+test set
    so that dummy columns line up.
    """
    combined = pd.concat({"train": train, "test": test}, names=["split", "row_id"])
    if genres_col not in combined.columns:
        # Nothing to do if there is no genres column at all.
        return train.copy(), test.copy()

    combined = combined.copy()
    combined[genres_col] = combined[genres_col].apply(_normalize_genre_cell)

    dummies = combined[genres_col].fillna("").str.get_dummies(sep="|")
    if dummies.shape[1] > 0:
        combined = pd.concat(
            [combined, dummies.add_prefix("genre_")],
            axis=1,
        )

    train_out = combined.xs("train").copy()
    test_out = combined.xs("test").copy()
    return train_out, test_out


# ---------- target encoding for categoricals ----------


def target_encode_single(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str,
    min_samples_leaf: int = 5,
    smoothing: float = 10.0,
    prefix: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Smoothed target encoding for one categorical column.

    Uses train-only stats and applies them to both train and test.
    """
    if prefix is None:
        prefix = col

    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(["mean", "count"])
    counts = agg["count"]
    means = agg["mean"]

    smoothing_factor = 1 / (1 + np.exp(-(counts - min_samples_leaf) / smoothing))
    encodings = global_mean * (1 - smoothing_factor) + means * smoothing_factor

    train_enc = train[col].map(encodings).fillna(global_mean)
    test_enc = test[col].map(encodings).fillna(global_mean)

    name = f"te_{prefix}"
    train_enc.name = name
    test_enc.name = name
    return train_enc, test_enc


def add_target_encodings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = TARGET_COL,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add target-encoded versions of selected categorical columns if present."""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FOR_TARGET_ENCODING

    train = train.copy()
    test = test.copy()

    for col in categorical_cols:
        if col in train.columns:
            tr_enc, te_enc = target_encode_single(train, test, col, target_col, prefix=col)
            train[tr_enc.name] = tr_enc
            test[te_enc.name] = te_enc

    return train, test


# ---------- feature selection ----------


def _select_feature_columns(train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (id_cols, feature_cols) for a training frame.

    ID columns are carried for error analysis only, not used for modeling.
    Feature columns are numeric, excluding the target.
    """
    id_cols_existing = [c for c in ID_COLS if c in train.columns]
    candidate_feature_cols = [
        c for c in train.columns
        if c not in id_cols_existing and c != TARGET_COL
    ]
    feature_cols = [
        c
        for c in candidate_feature_cols
        if pd.api.types.is_numeric_dtype(train[c])
    ]
    return id_cols_existing, feature_cols


def _drop_rows_with_missing(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Drop rows with NaNs in feature or target columns."""
    df = df.copy()
    cols_to_check = list(feature_cols)
    if target_col in df.columns:
        cols_to_check.append(target_col)
    if cols_to_check:
        df = df.dropna(subset=cols_to_check)
    return df


def build_features_for_split(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the feature matrices for one time-based split.

    Steps:
      - basic cleaning
      - simple numeric features
      - genre dummies with aligned columns
      - optional target encodings
      - keep ID columns + numeric features + target
      - drop rows with missing values in features/target
    """
    # Cleaning and basic numeric features
    train = add_numeric_features(basic_clean(train_raw))
    test = add_numeric_features(basic_clean(test_raw))

    # Genres
    train, test = add_genre_dummies(train, test)

    # Categorical target encodings (if those columns exist)
    train, test = add_target_encodings(train, test)

    # Final feature selection
    id_cols, feature_cols = _select_feature_columns(train)
    train = _drop_rows_with_missing(train, feature_cols, TARGET_COL)
    test = _drop_rows_with_missing(test, feature_cols, TARGET_COL)

    train_out = train[id_cols + feature_cols + [TARGET_COL]]
    test_out = test[id_cols + feature_cols + [TARGET_COL]]
    return train_out, test_out


# ---------- time-based splitting ----------


def _iter_time_splits(
    df: pd.DataFrame,
    gap_years: int,
    min_train_size: int,
    min_test_size: int,
) -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yield (cutoff_year, train_raw, test_raw) pairs.

    For each cutoff year T:
      - train: YEAR_COL <= T
      - test:  YEAR_COL in [T+1, T+gap_years]
    """
    if YEAR_COL not in df.columns:
        raise RuntimeError(f"Year column not found in data (expected '{YEAR_COL}')")

    years = sorted(df[YEAR_COL].dropna().unique())
    for cutoff in years:
        train_mask = df[YEAR_COL] <= cutoff
        test_mask = df[YEAR_COL].between(cutoff + 1, cutoff + gap_years)

        if train_mask.sum() < min_train_size or test_mask.sum() < min_test_size:
            continue

        yield int(cutoff), df.loc[train_mask].copy(), df.loc[test_mask].copy()


def generate_time_based_splits(
    movies_path: Path = RAW_MOVIES_PATH,
    gap_years: int = 2,
    min_train_size: int = 300,
    min_test_size: int = 80,
) -> None:
    """
    Create multiple train/test CSVs where test years strictly follow train years.

    Saves:
      - train_until_<cutoff>.csv
      - test_<start>_to_<end>.csv
      - time_splits_summary.csv with metadata for all splits
    """
    ensure_dirs()

    df_raw = pd.read_csv(movies_path)
    df_raw = basic_clean(df_raw)

    all_splits_meta = []

    for cutoff, train_raw, test_raw in _iter_time_splits(
        df_raw,
        gap_years=gap_years,
        min_train_size=min_train_size,
        min_test_size=min_test_size,
    ):
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