
# preprocess.py
# Script to preprocess TMDB 5000 dataset and generate pickle files

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk
import logging
import argparse
import os
import zipfile
import subprocess
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download NLTK punkt dataset if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt dataset already downloaded.")
    except LookupError:
        logger.info("Downloading NLTK punkt dataset...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt dataset downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading NLTK punkt dataset: {str(e)}")
            raise

ps = PorterStemmer()

def stem(text):
    y = [ps.stem(i) for i in text.split()]
    return " ".join(y)

def download_data():
    """Download and extract TMDB 5000 dataset from Kaggle if not present."""
    if not os.path.exists('data/movies.csv') or not os.path.exists('data/credits.csv'):
        logger.info("Downloading TMDB 5000 dataset from Kaggle...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", "tmdb/tmdb-movie-metadata", "-p", "data/"], check=True)
            with zipfile.ZipFile('data/tmdb-movie-metadata.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')
            # Rename files to match expected paths
            os.rename('data/tmdb_5000_movies.csv', 'data/movies.csv')
            os.rename('data/tmdb_5000_credits.csv', 'data/credits.csv')
            os.remove('data/tmdb-movie-metadata.zip')
            logger.info("Dataset downloaded and extracted.")
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise

def load_and_process_data(movies_path='data/movies.csv', credits_path='data/credits.csv'):
    """Load, merge, and process TMDB data to create tags."""
    try:
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
        
        # Remove duplicates
        movies.drop_duplicates(subset='id', keep='first', inplace=True)
        credits.drop_duplicates(subset='movie_id', keep='first', inplace=True)
        
        # Merge on movie_id (id in movies.csv, movie_id in credits.csv)
        df = movies.merge(credits, left_on='id', right_on='movie_id', how='left', suffixes=('', '_credits'))
        
        # Create tags
        def create_tags(row):
            # Parse JSON-like columns
            genres = ' '.join([g['name'] for g in json.loads(row['genres'])]) if pd.notna(row['genres']) else ''
            keywords = ' '.join([k['name'] for k in json.loads(row['keywords'])]) if pd.notna(row['keywords']) else ''
            cast = ' '.join([c['name'].replace(' ', '') for c in json.loads(row['cast'])[:3]]) if pd.notna(row['cast']) else ''
            director = next((c['name'].replace(' ', '') for c in json.loads(row['crew']) if c['job'] == 'Director'), '') if pd.notna(row['crew']) else ''
            overview = str(row['overview']) if pd.notna(row['overview']) else ''
            return f"{overview} {genres} {keywords} {cast} {director}"
        
        df['tags'] = df.apply(create_tags, axis=1)
        df = df[['id', 'title', 'genres', 'tags']]  # Keep genres for display
        df['title'] = df['title'].str.strip()  # Clean titles
        
        # Preprocess tags
        df['tags'] = df['tags'].apply(lambda x: x.lower())
        df['tags'] = df['tags'].apply(stem)
        
        logger.info("Data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def generate_vectors_and_similarity(df):
    """Vectorize tags and compute similarity."""
    try:
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        vectors = tfidf.fit_transform(df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        logger.info("Vectors and similarity matrix generated.")
        return similarity
    except Exception as e:
        logger.error(f"Error generating vectors/similarity: {str(e)}")
        raise

def save_data(df, similarity):
    try:
        df.to_pickle('movies.pkl')
        np.save('similarity.npy', similarity)
        logger.info("Data saved to 'movies.pkl' and 'similarity.npy'.")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TMDB 5000 data for recommender.")
    parser.add_argument('--download', action='store_true', help="Download dataset if not present")
    args = parser.parse_args()
    
    download_nltk_data()
    if args.download:
        download_data()
    
    df = load_and_process_data()
    similarity = generate_vectors_and_similarity(df)
    save_data(df, similarity)
