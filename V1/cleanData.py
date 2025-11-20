import pandas as pd

# Load the dataset
df = pd.read_csv("imdb_movies_clean_50.csv")

# Define the columns to keep
keep_cols = [
    "director_name",
    "duration",
    "director_facebook_likes",
    "actor_3_facebook_likes",
    "actor_2_name",
    "actor_1_facebook_likes",
    "gross",
    "genres",
    "actor_1_name",
    "cast_total_facebook_likes",
    "actor_3_name",
    "budget",
    "title_year",
    "actor_2_facebook_likes",
    "imdb_score"
]

# Keep only those columns
df_clean = df[keep_cols]

# Optionally: remove duplicates and drop rows with all missing values
df_clean = df_clean.drop_duplicates().dropna(how="all")

# Save the cleaned dataset
df_clean.to_csv("imdb_movies_cleaned.csv", index=False)

print("✅ Cleaned dataset saved as imdb_movies_clean.csv")
