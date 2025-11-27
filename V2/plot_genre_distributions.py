# plot_imdb_by_genre.py

import pandas as pd
from pathlib import Path

from data_prep_time_splits import SPLITS_DIR, TARGET_COL
from evaluation_plots import plot_imdb_histograms_by_genre


def load_all_split_data() -> pd.DataFrame:
    """
    Load all train/test CSVs from SPLITS_DIR and concatenate them
    into a single DataFrame for genre analysis.
    """
    meta_path = SPLITS_DIR / "time_splits_summary.csv"
    meta = pd.read_csv(meta_path)

    dfs = []
    for _, row in meta.iterrows():
        train_path = Path(row["train_path"])
        test_path = Path(row["test_path"])
        dfs.append(pd.read_csv(train_path))
        dfs.append(pd.read_csv(test_path))

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def main():
    df = load_all_split_data()
    plot_imdb_histograms_by_genre(df, target_col=TARGET_COL)


if __name__ == "__main__":
    main()
