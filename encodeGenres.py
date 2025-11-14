from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

df = pd.read_csv("imdb_movies_clean.csv")

df["genres_list"] = df["genres"].str.split("|")

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(df["genres_list"])

print(X)
print(mlb.classes_)