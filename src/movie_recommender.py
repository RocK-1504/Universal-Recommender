import pickle
import pandas as pd

movies = pd.read_csv("data/movielens/movie.csv")

with open("models/similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
with open("models/similaritySVD.pkl", "rb") as f:
    similaritySVD = pickle.load(f)
with open("models/movie_indices.pkl", "rb") as f:
    movie_indices = pickle.load(f)
with open("models/title_to_id.pkl", "rb") as f:
    title_to_id = pickle.load(f)
with open("models/movie_matrix_index.pkl", "rb") as f:
    movie_index = pickle.load(f)
with open("models/genre_indices.pkl", "rb") as f:
    genre_indices = pickle.load(f)
with open("models/genre_similarity.pkl", "rb") as f:
    genre_similarity = pickle.load(f)
with open("models/tag_similarity.pkl","rb") as f:
    tag_similarity = pickle.load(f)
with open("models/tag_indices.pkl","rb") as f:
    tag_indices = pickle.load(f)

id_to_title = dict(zip(movies["movieId"], movies["title"]))

def recommend(movie_title):

    movie_id = title_to_id.get(movie_title)
    if movie_id is None:
        return []

    idx = movie_indices[movie_id]
    genre_idx = genre_indices[movie_id]
    tag_idx = tag_indices[movie_id]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_movies = scores[1:80]

    reranked = []
    for i, _ in top_movies:

        svd_score = similaritySVD[idx][i]
        cos_score = similarity[idx][i]
        movie_id_candidate = movie_index[i]
        genre_score = genre_similarity[genre_idx][genre_indices[movie_id_candidate]]
        tag_score = tag_similarity[tag_idx][tag_indices[movie_id_candidate]]

        final_score = (
            0.25 * cos_score +
            0.25 * svd_score +
            0.25 * genre_score+
            0.25 * tag_score
        )

        reranked.append((i, final_score))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

    movie_ids = [movie_index[i[0]] for i in reranked[:5]]

    return [id_to_title[mid] for mid in movie_ids]
