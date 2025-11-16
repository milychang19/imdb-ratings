from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

df = pd.read_csv("imdb_movies_clean.csv")

df["genres_list"] = df["genres"].str.split("|")

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(df["genres_list"])

print(X)
print(mlb.classes_)


# Lines 16-## to get director scores for encoding (uncomplete, untested)

#to normilize
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['imdb_norm', 'likes_norm', 'revenue_norm']] = scaler.fit_transform(
    df[['imdb_score', 'director_facebook_likes', 'revenue']]
)#we might want to drop the non normilizaed columns

w_imdb = 0.5
w_rev  = 0.3
w_like = 0.2

#get score for each movie based on imdb score, revenue, and facebook likes
df['director_score'] = (
    df['imdb_norm']  * w_imdb +
    df['revenue_norm'] * w_rev +
    df['likes_norm'] * w_like
)

#assigns average movie performence to each director based on the average of all of their movies in the dataset
director_scores = df.groupby('director_name')['director_score'].mean().rename("final_director_score")

df = df.merge(director_scores, on='director_name', how='left')#merge column into dataset
df = df.rename(columns={'director_name': 'director'})#rename column

df = df.drop(columns=['director_score'])#drop the per movie score column after we are done
