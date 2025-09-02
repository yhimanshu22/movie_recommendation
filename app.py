# app.py
# Advanced Movie Recommender with Netflix/Amazon-style Features

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import json
import subprocess
import requests

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------ Config ------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "your_tmdb_api_key_here")  # Set via env variable

# ------------------ Preprocessing Check ------------------
def ensure_data():
    if not (os.path.exists("movies.pkl") and os.path.exists("similarity.npy")):
        st.warning("Pickle files not found. Running preprocessing...")
        try:
            subprocess.run(["python", "preprocess.py", "--download"], check=True)
            st.success("Preprocessing complete. Data ready!")
        except Exception as e:
            st.error(f"Preprocessing failed: {str(e)}")
            st.stop()

ensure_data()

# ------------------ Load Data ------------------
@st.cache_resource
def load_data():
    df = pd.read_pickle("movies.pkl")
    similarity = np.load("similarity.npy")
    return df, similarity

# ------------------ Helpers ------------------
def parse_genres(genres_raw):
    try:
        return ", ".join([g["name"] for g in json.loads(genres_raw)])
    except Exception:
        return str(genres_raw)

def fetch_poster(movie_title):
    """Fetch poster URL from TMDB API."""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url).json()
        poster_path = response["results"][0]["poster_path"]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None

def fetch_trailer(movie_title):
    """Fetch YouTube trailer link from TMDB API."""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url).json()
        movie_id = response["results"][0]["id"]

        trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        trailer_data = requests.get(trailer_url).json()

        for video in trailer_data["results"]:
            if video["site"] == "YouTube" and video["type"] == "Trailer":
                return f"https://www.youtube.com/watch?v={video['key']}"
    except:
        return None

def recommend(movie_title, df, similarity, top_n=5):
    try:
        movie_index = df[df["title"].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(
            list(enumerate(distances)), reverse=True, key=lambda x: x[1]
        )[1 : top_n + 1]
        return [(df.iloc[i[0]].title, df.iloc[i[0]].genres) for i in movies_list]
    except IndexError:
        return [("Movie not found in the dataset", "")]
    except Exception as e:
        return [(f"Error: {str(e)}", "")]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="wide")
st.title("🎬 Movie Recommender System")
st.markdown(
    """
This app recommends movies using the TMDB 5000 dataset + TMDB API.  
Now with Netflix-style UI: Posters, Trailers, Watchlist, Trending, and Filters. 🚀
"""
)

# Load data
df, similarity = load_data()

# Sidebar options
with st.sidebar:
    st.header("⚙️ Options")
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    st.markdown("---")

    st.subheader("Filters")
    min_year, max_year = st.slider("Release Year", 1950, 2025, (2000, 2020))
    genre_filter = st.text_input("Filter by Genre (e.g., Action, Comedy)")
    st.markdown("---")

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    st.subheader("📌 Watchlist")
    if st.session_state.watchlist:
        for movie in st.session_state.watchlist:
            st.markdown(f"- {movie}")
    else:
        st.caption("No movies added yet.")

# Search UI
st.subheader("🔍 Find a Movie")
search_query = st.text_input("Type movie title (e.g., Avatar)")

movie_titles = sorted(df["title"].unique())
selected_movie = st.selectbox("Or choose from list:", ["Select a movie..."] + movie_titles)

# Use search query if entered
movie_input = (
    search_query.strip()
    if search_query
    else (selected_movie if selected_movie != "Select a movie..." else None)
)

# Trending Movies
st.subheader("🔥 Trending Movies")
trending = df.sample(5).title.tolist()
cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        poster = fetch_poster(trending[i])
        if poster:
            st.image(poster, use_column_width=True)
        st.caption(trending[i])

# Recommendation Section
if st.button("Get Recommendations"):
    if movie_input:
        with st.spinner("Generating recommendations..."):
            recommendations = recommend(movie_input, df, similarity, top_n=top_n)

        st.subheader(f"Top {top_n} Recommendations for '{movie_input}':")

        if recommendations and not recommendations[0][0].startswith(
            ("Movie not found", "Error")
        ):
            cols = st.columns(2)
            for i, (movie, genres) in enumerate(recommendations, 1):
                with cols[i % 2]:
                    st.markdown(f"### {i}. {movie}")
                    st.caption(f"Genres: {parse_genres(genres)}")

                    poster = fetch_poster(movie)
                    if poster:
                        st.image(poster, width=200)

                    trailer = fetch_trailer(movie)
                    if trailer:
                        st.video(trailer)

                    if st.button(f"➕ Add {movie} to Watchlist", key=f"wl_{movie}"):
                        st.session_state.watchlist.append(movie)
        else:
            st.error(recommendations[0][0])
    else:
        st.warning("Please search or select a movie.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data: TMDB 5000 + TMDB API | Features: Posters, Trailers, Watchlist, Filters")
