import pandas as pd

# Load cleaned dataset
df = pd.read_csv("imdb_movies_clean.csv")

# Identify numeric columns automatically
numeric_cols = df.select_dtypes(include=["number"]).columns

# Fill missing numeric values with column mean
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

# Optionally, fill missing text columns with "Unknown"
# df = df.fillna("Unknown")

# Save the final cleaned dataset
df.to_csv("imdb_movies_final.csv", index=False)

print("✅ Missing numeric values filled with column mean.")
print(f"Remaining missing values:\n{df.isna().sum()}")
