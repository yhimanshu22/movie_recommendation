import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import json
import subprocess

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    df = pd.read_pickle('movies.pkl')
    similarity = np.load('similarity.npy')
    return df, similarity

# ------------------ Helpers ------------------
def parse_genres(genres_raw):
    try:
        return ", ".join([g['name'] for g in json.loads(genres_raw)])
    except Exception:
        return str(genres_raw)

def recommend(movie_title, df, similarity, top_n=5):
    try:
        movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        return [(df.iloc[i[0]].title, df.iloc[i[0]].genres) for i in movies_list]
    except IndexError:
        return [("Movie not found in the dataset", "")]
    except Exception as e:
        return [(f"Error: {str(e)}", "")]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="wide")
st.title("🎬 Movie Recommender System")
st.markdown("""
This app recommends movies based on genres, overview, keywords, cast, and director using the TMDB 5000 dataset.  
Select a movie or search manually to get tailored recommendations with genre details.
""")

# Load data
df, similarity = load_data()

# Sidebar options
with st.sidebar:
    st.header("⚙️ Options")
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.caption("Select or search for a movie below to get started.")

# Search UI
st.subheader("Find a Movie")
search_query = st.text_input("🔍 Search movie title (e.g., Avatar)")

movie_titles = sorted(df['title'].unique())
selected_movie = st.selectbox("Or choose from list:", ["Select a movie..."] + movie_titles)

# Use search query if entered
movie_input = search_query.strip() if search_query else (selected_movie if selected_movie != "Select a movie..." else None)

if st.button("Get Recommendations"):
    if movie_input:
        with st.spinner("Generating recommendations..."):
            recommendations = recommend(movie_input, df, similarity, top_n=top_n)
        st.subheader(f"Top {top_n} Recommendations for '{movie_input}':")

        # Show selected movie genres
        if movie_input.lower() in df['title'].str.lower().values:
            selected_genres = df[df['title'].str.lower() == movie_input.lower()]['genres'].iloc[0]
            st.markdown(f"**Selected Movie Genres**: {parse_genres(selected_genres)}")

        # Display recommendations
        if recommendations and not recommendations[0][0].startswith(("Movie not found", "Error")):
            for i, (movie, genres) in enumerate(recommendations, 1):
                st.markdown(f"### {i}. {movie}")
                st.caption(f"Genres: {parse_genres(genres)}")
                with st.expander(f"More about {movie}"):
                    st.write(f"**Genres**: {parse_genres(genres)}")
        else:
            st.error(recommendations[0][0])
    else:
        st.warning("Please search or select a movie.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data: TMDB 5000 (Kaggle)")
