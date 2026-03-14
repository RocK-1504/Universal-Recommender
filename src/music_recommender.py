import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack 

features = [
    'acousticness',
    'danceability',
    'energy',
    'instrumentalness',
    'liveness',
    'loudness',
    'speechiness',
    'tempo',
    'valence',
    'key',
    'mode'
]
df = pd.read_csv("data/spotify/data.csv")
df["song"] = df["name"] + " - " + df["artists"]
df["artists_text"] = df["artists"].str.replace("[", "", regex=False)\
.str.replace("]", "", regex=False)\
.str.replace("'", "", regex=False)

vectorizer = TfidfVectorizer()
artist_matrix = vectorizer.fit_transform(df["artists_text"])

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

combined_features = hstack([artist_matrix,X_pca]).tocsr()

model = NearestNeighbors(
    n_neighbors=100,
    metric="cosine"
)
model.fit(combined_features)

def recommend(song_name):

    match = df[df["song"] == song_name]

    if match.empty:
        return pd.DataFrame(columns=["name","artists","popularity","year"])

    song = match.iloc[0]
    idx = match.index[0]

    song_year = song["year"]
    song_energy = song["energy"]
    song_tempo = song["tempo"]
    base_name = song["name"].split(" - ")[0]

    distances, indices = model.kneighbors(combined_features[idx:idx+1])

    rec_songs = df.iloc[indices[0]][["name","artists","popularity","year","energy","tempo"]]

    rec_songs = rec_songs[
        (rec_songs["year"] >= song_year-10) &
        (rec_songs["year"] <= song_year+10)
    ]

    rec_songs = rec_songs[
        (rec_songs["energy"] >= song_energy-0.4) &
        (rec_songs["energy"] <= song_energy+0.4)
    ]

    rec_songs = rec_songs[
        (rec_songs["tempo"] >= song_tempo-15) &
        (rec_songs["tempo"] <= song_tempo+15)
    ]

    rec_songs = rec_songs[
    ~rec_songs["name"].str.startswith(base_name)
    ]

    rec_songs = rec_songs.sort_values("popularity", ascending=False)
    return rec_songs[["name", "artists", "popularity", "year"]].head(5)


