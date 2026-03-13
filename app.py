import streamlit as st
import pandas as pd
import requests

@st.cache_resource
def load_recommender():
    from src.movie_recommender import recommend
    return recommend

recommend = load_recommender()

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Smart Movie Recommender")
st.caption("Hybrid recommendation system (Cosine Similarity + SVD)")

st.markdown("---")

@st.cache_data
def load_data():
    movies = pd.read_csv("data/movielens/movie.csv")
    links = pd.read_csv("data/movielens/link.csv")

    links = links.dropna(subset=["imdbId"])
    links["imdbId"] = links["imdbId"].astype(int)

    movies = movies[movies["movieId"].isin(links["movieId"])]

    imdb_map = dict(zip(links["movieId"], links["imdbId"]))

    return movies, imdb_map

movies, imdb_map = load_data()

movie_titles = movies["title"].tolist()

title_to_movieid = dict(zip(movies["title"], movies["movieId"]))


OMDB_API_KEY = "YOUR_API_KEY"    #Enter your api key

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_data(imdb_id):

    imdb_id_str = f"tt{str(imdb_id).zfill(7)}"

    url = f"https://www.omdbapi.com/?i={imdb_id_str}&apikey={OMDB_API_KEY}"

    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        if data.get("Response") == "True":
            return data

    except Exception:
        return None

    return None


def get_imdb_id(movie_id):
    return imdb_map.get(movie_id)

selected_movie = st.selectbox(
    "🎥 Search for a movie",
    options=movie_titles,
    index=None,
    placeholder="Start typing a movie name...",
)

if selected_movie and st.button("Recommend Movies 🍿"):
    st.session_state["run_movie"] = selected_movie
    st.session_state.pop("recommendations", None)


if "run_movie" in st.session_state:

    with st.spinner("Fetching recommendations..."):

        if "recommendations" not in st.session_state:
            st.session_state["recommendations"] = recommend(st.session_state["run_movie"])

        recommendations = st.session_state["recommendations"]

        st.markdown(f"#### Because you liked **{st.session_state['run_movie']}**:")

        cols = st.columns(5)

        for i, movie in enumerate(recommendations):

            movie_id = title_to_movieid.get(movie)

            imdb_id = get_imdb_id(movie_id)

            movie_data = None
            if imdb_id:
                movie_data = fetch_movie_data(imdb_id)

            with cols[i]:

                if movie_data:

                    poster = movie_data.get("Poster")

                    if poster and poster != "N/A":
                        st.image(poster, use_container_width=True)

                    st.markdown(
                        f"""
                        **{movie}**  
                        ⭐ {movie_data.get('imdbRating', 'N/A')}  
                        📅 {movie_data.get('Year', 'N/A')}
                        """
                    )

                else:
                    st.markdown(f"**{movie}**")
