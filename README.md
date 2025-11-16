# imdb-ratings
Data science: Building regression model that predicts movie ratings
## Data
- **movie_metadata.csv**: the raw 5000 movie dataset from IMDB
- **imdb_movie_clean.csv**: clean duplicates and remove extra info
- **imdb_movie_final.csv**: final preprocessed dataset for modeling
- **director_summary.csv**: summarize director's experience (film counts, avg imdb score, avg gross)
- **actors_summary.csv**: summarize actor's experience (film counts, avg imdb score, avg gross)
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