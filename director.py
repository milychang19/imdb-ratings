import pandas as pd

# Load your cleaned IMDb dataset
df = pd.read_csv("imdb_movies_final.csv")

# Drop rows missing director info
df = df.dropna(subset=["director_name"])

# --- DIRECTOR STATS ---
director_df = df.groupby("director_name").agg(
    director_movie_count=("title_year", "count"),     # count how many movies per director
    director_avg_imdb_score=("imdb_score", "mean"),   # average IMDb score
    director_avg_gross=("gross", "mean")              # average gross revenue
).reset_index()

# Save the director summary
director_df.to_csv("director_summary.csv", index=False)

print("✅ Director summary saved as director_summary.csv")
