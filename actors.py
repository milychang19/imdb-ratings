import pandas as pd

# Load your cleaned IMDb dataset
try:
    df = pd.read_csv("imdb_movies_final.csv")
except FileNotFoundError:
    print("Error: 'imdb_movies_final.csv' not found.")
    # Exit or handle the error appropriately
    exit()

# --- Combined Actor Summary Logic ---

# 1. Define the columns we're interested in
actor_cols = ['actor_1_name', 'actor_2_name', 'actor_3_name']
metric_cols = ['imdb_score', 'gross']

# 2. "Melt" the DataFrame
# This stacks the three actor columns into a single 'actor_name' column.
# Each movie will now have up to 3 rows, one for each listed actor.
melted_df = df[actor_cols + metric_cols].melt(
    id_vars=metric_cols,           # Columns to keep as-is
    value_vars=actor_cols,         # Columns to "unpivot" or "stack"
    var_name='billing_position',   # We won't use this, but melt creates it
    value_name='actor_name'        # The new column with all actor names
)

# 3. Clean the melted data
# Drop rows where the actor name was missing (e.g., movies with only 1 or 2 actors)
cleaned_df = melted_df.dropna(subset=['actor_name'])

# 4. Group by the combined 'actor_name' column and aggregate
# Now we can perform the exact same aggregation as your original code,
# but on the combined data.
summary = (
    cleaned_df.groupby('actor_name')
    .agg(
        # Count all movie appearances for this actor
        actor_movie_count=('actor_name', 'count'),
        # Calculate mean score (mean() automatically ignores NaNs)
        actor_avg_imdb_score=('imdb_score', 'mean'),
        # Calculate mean gross (mean() automatically ignores NaNs)
        actor_avg_gross=('gross', 'mean'),
    )
    .reset_index() # Turn 'actor_name' from an index back into a column
    .sort_values(
        ['actor_movie_count', 'actor_avg_imdb_score'],
        ascending=[False, False]
    )
)

# 5. Save the single, combined summary file
output_filename = "actors_summary.csv"
summary.to_csv(output_filename, index=False)

print(f"✅ Saved {output_filename} with {len(summary)} actors.")
print("Combined actor summary generated successfully.")