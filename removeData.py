import pandas as pd

# Load your latest cleaned dataset
df = pd.read_csv("imdb_movies_final.csv")

# Drop rows where director_name is missing
df = df.dropna(subset=["director_name"])

# Reset index (optional)
df = df.reset_index(drop=True)

# Save the cleaned version
df.to_csv("imdb_movies_final_no_missing_director.csv", index=False)

print(f"✅ Rows with missing director removed. Remaining rows: {len(df)}")
