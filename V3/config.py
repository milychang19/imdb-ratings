# config.py

from pathlib import Path
from typing import List

# ---------- Paths ----------

DATA_DIR = Path("data")
RAW_MOVIES_PATH = DATA_DIR / "movie_metadata.csv"
SPLITS_DIR = DATA_DIR / "time_splits"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

# ---------- Columns ----------

TARGET_COL = "vote_average"
DATE_COL = "release_date"
YEAR_COL = "release_year"

GENRES_COL = "genres"
BUDGET_COL = "budget"
GROSS_COL = "revenue"

NUMERIC_BASE_COLS: List[str] = [
    "runtime",
    "vote_count",
    "revenue",
    "budget",
    "popularity",
]

# ID / metadata columns to keep for analysis, not as features
ID_COLS: List[str] = [
    "id",
    "title",
    DATE_COL,
    GENRES_COL,
    "original_language",
    "original_title",
    "imdb_id",
]

# Categorical columns we optionally target-encode if present
CATEGORICAL_FOR_TARGET_ENCODING: List[str] = [
    "language",
    "country",
    "content_rating",
    "color",
]
