import pandas as pd
from pathlib import Path

# --- filenames ---
src = "imdb_movies_final.csv"   # note: file is **imdb** not "imbd"
bak = "imdb_movies_final.backup.csv"

# Load
df = pd.read_csv(src)

# Make a backup (once)
if not Path(bak).exists():
    df.to_csv(bak, index=False)

# 1) Compute per-movie average cast likes (mean of available values)
like_cols = ["actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes"]
existing_like_cols = [c for c in like_cols if c in df.columns]

if existing_like_cols:
    df["avg_cast_likes"] = df[existing_like_cols].mean(axis=1)  # skips NaNs per row
else:
    # If somehow those columns don't exist, create the field as NaN
    df["avg_cast_likes"] = pd.NA

# 2) Drop the individual like columns
df = df.drop(columns=existing_like_cols, errors="ignore")

# Save back to the same filename
df.to_csv(src, index=False)

print("✅ Added avg_cast_likes and removed per-actor like columns.")
print(f"Columns now: {list(df.columns)}")
