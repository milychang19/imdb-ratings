# imdb-ratings
Data science: Building regression model that predicts movie ratings

## Data
- movie_metadata.csv: The raw IMDb dataset containing ~5000 movie entries with original columns (unprocessed).
- imdb_movie_clean.csv: Cleaned version of the raw data — duplicates and rows with excessive missing values removed. Only key columns are kept.
- imdb_movie_final.csv: The final preprocessed dataset ready for analysis and modeling. Includes imputed numeric means, avg_cast_likes, and both cast popularity metrics.
- director_summary.csv: Summarized director-level data with each director’s total movie count, average IMDb rating, and average gross revenue.
- actors_summary.csv: Summarized actor-level data showing each actor’s total number of films, average IMDb rating, and average gross revenue.

## Data Cleaning
### cleanData.py
- Define columns to keep from the raw CSV
- Removes duplicate rows
- Saves the result to imdb_movies_cleaned.csv
### addMean.py
- Drops rows with more than 50% missing values across the kept columns
- Fills missing values in numeric columns with the column mean
- Saves the result (imdb_movies_final.csv)
### facebookLikes.py
- Computes avg_cast_likes of lead actors
### director.py
- Drops rows with missing director_name
- Builds a director summary CSV (director_summary.csv) with movie count, avg imdb score, and avg gross
### actors.py
- Drops rows with missing actor names per position
- Builds a actor summary CSV (actors_summary.csv) with movie count, avg imdb score, and avg gross