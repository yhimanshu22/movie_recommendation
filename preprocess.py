import os
import json
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Global Paths
data_dir = Path("data")
movies_pkl = Path("movies.pkl")
similarity_npy = Path("similarity.npy")

ps = PorterStemmer()


def safe_json_loads(x, default="[]"):
    try:
        return json.loads(x) if pd.notna(x) else []
    except Exception:
        return json.loads(default)


def create_tags(row):
    genres = " ".join([g["name"] for g in safe_json_loads(row["genres"])])
    keywords = " ".join([k["name"] for k in safe_json_loads(row["keywords"])])
    cast = " ".join([c["name"].replace(" ", "") for c in safe_json_loads(row["cast"])[:3]])
    director = next(
        (c["name"].replace(" ", "") for c in safe_json_loads(row["crew"]) if c["job"] == "Director"),
        "",
    )
    overview = str(row["overview"]) if pd.notna(row["overview"]) else ""
    return f"{overview} {genres} {keywords} {cast} {director}".strip()


def stem(text: str) -> str:
    return " ".join([ps.stem(word) for word in text.split()])


def download_dataset():
    """
    Tries to fetch dataset.
    If Kaggle CLI isn't available (e.g., Streamlit Cloud), fallback to a direct link.
    """
    if data_dir.exists():
        print("✅ Dataset already exists.")
        return

    try:
        print("📥 Downloading dataset from Kaggle...")
        os.system("kaggle datasets download -d tmdb/tmdb-movie-metadata -p ./")
        os.system("unzip -q tmdb-movie-metadata.zip -d data")
        os.remove("tmdb-movie-metadata.zip")
        print("✅ Kaggle dataset downloaded.")
    except Exception as e:
        print("⚠️ Kaggle download failed, trying fallback...")
        # Example fallback: Replace with your own uploaded dataset link
        url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
        r = requests.get(url)
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "movies.csv", "wb") as f:
            f.write(r.content)
        print("✅ Fallback dataset downloaded.")


def preprocess():
    download_dataset()

    # Load CSVs
    movies = pd.read_csv(data_dir / "movies.csv")
    credits = pd.read_csv(data_dir / "credits.csv")

    # Merge datasets
    df = movies.merge(credits, on="id")

    # Select features
    df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    # Drop NaNs
    df.dropna(inplace=True)

    # Create tags
    tqdm.pandas(desc="🔄 Creating tags")
    df["tags"] = df.progress_apply(create_tags, axis=1)

    # Process tags
    df["tags"] = df["tags"].str.lower().apply(stem)

    # Vectorize
    print("⚡ Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"]).toarray()

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(vectors)

    # Save
    with open(movies_pkl, "wb") as f:
        pickle.dump(df[["id", "title", "tags"]], f)

    np.save(similarity_npy, similarity)

    print("✅ Preprocessing complete. Files saved:")
    print(f"  - {movies_pkl}")
    print(f"  - {similarity_npy}")


if __name__ == "__main__":
    preprocess()
