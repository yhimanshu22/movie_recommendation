
# app.py
# Advanced Streamlit app for TMDB 5000-based movie recommendations

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to load data and similarity matrix
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('movies.pkl')
        similarity = np.load('similarity.npy')
        logger.info("Data and similarity matrix loaded successfully.")
        return df, similarity
    except FileNotFoundError:
        logger.error("Pickle files not found. Please run preprocess.py first.")
        st.error("Pickle files not found. Please run preprocess.py first.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Recommendation function
def recommend(movie_title, df, similarity, top_n=5):
    try:
        movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        recommended_movies = [(df.iloc[i[0]].title, df.iloc[i[0]].genres) for i in movies_list]
        return recommended_movies
    except IndexError:
        return [("Movie not found in the dataset", "")]
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return [(f"Error: {str(e)}", "")]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="wide")
st.title("🎬 Industry-Standard Movie Recommender System")
st.markdown("""
This app recommends movies based on genres, overview, keywords, cast, and director using the TMDB 5000 dataset.  
Select a movie from the dropdown or type to search, then get tailored recommendations with genre details.
""")

# Load data
df, similarity = load_data()

if df is not None and similarity is not None:
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Options")
        top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
        st.markdown("---")
        st.caption("Select or search for a movie below to get started.")

    # Create sorted list of movie titles for dropdown
    movie_titles = sorted(df['title'].str.lower().unique())
    movie_titles = [title.title() for title in movie_titles]  # Capitalize for display
    movie_titles.insert(0, "Select a movie...")  # Placeholder option

    # Main input: Searchable dropdown
    st.subheader("Select or Search for a Movie")
    selected_movie = st.selectbox(
        "Choose a movie title:",
        options=movie_titles,
        index=0,
        help="Type to search or select a movie from the list (e.g., 'Avatar')."
    )

    # Button to trigger recommendations
    if st.button("Get Recommendations", key="recommend_button"):
        if selected_movie != "Select a movie...":
            with st.spinner("Generating recommendations..."):
                recommendations = recommend(selected_movie.lower(), df, similarity, top_n=top_n)
            st.subheader(f"Top {top_n} Recommended Movies for '{selected_movie}':")
            
            # Display selected movie's genres
            selected_movie_lower = selected_movie.lower()
            if selected_movie_lower in df['title'].str.lower().values:
                selected_genres = df[df['title'].str.lower() == selected_movie_lower]['genres'].iloc[0]
                try:
                    genres_list = [g['name'] for g in json.loads(selected_genres)]
                    st.markdown(f"**Selected Movie Genres**: {', '.join(genres_list)}")
                except:
                    st.markdown(f"**Selected Movie Genres**: {selected_genres}")
            
            # Display recommendations with genres
            if recommendations and recommendations[0][0] != "Movie not found in the dataset" and not recommendations[0][0].startswith("Error"):
                st.markdown("### Recommendations")
                col1, col2 = st.columns([2, 1])
                for i, (movie, genres) in enumerate(recommendations, 1):
                    with col1:
                        st.markdown(f"**{i}. {movie}**")
                    with col2:
                        try:
                            genres_list = [g['name'] for g in json.loads(genres)]
                            st.markdown(f"Genres: {', '.join(genres_list)}")
                        except:
                            st.markdown(f"Genres: {genres}")
                    with st.expander(f"Details for {movie}"):
                        try:
                            genres_list = [g['name'] for g in json.loads(genres)]
                            st.write(f"**Genres**: {', '.join(genres_list)}")
                        except:
                            st.write(f"**Genres**: {genres}")
            else:
                st.error(recommendations[0][0])
        else:
            st.warning("Please select a movie from the dropdown.")
else:
    st.stop()

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Data sourced from Kaggle/TMDB 5000. For issues, contact support or check logs.")
