# Item-based collaborative filtering using cosine similarity on user-item matrix.
# Works with ratings.csv (columns: userId, movieId, rating)

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_ratings(path: str = "data/ratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep essential columns
    return df[["userId", "movieId", "rating"]].dropna()

def build_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    # pivot -> rows=user, cols=movieId, values=rating
    return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0.0)

def cosine_sim_matrix(M: np.ndarray) -> np.ndarray:
    # Normalize rows to unit norm, then dot
    norm = np.linalg.norm(M, axis=0, keepdims=True) + 1e-9
    X = M / norm
    return X.T @ X  # item-item similarity

def fit_item_cf(ratings_path: str = "data/ratings.csv") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ratings = load_ratings(ratings_path)
    mat = build_item_matrix(ratings)          # users x items
    sim = cosine_sim_matrix(mat.values)       # items x items
    item_ids = mat.columns.to_numpy()
    return mat, sim, item_ids

def recommend_for_user(
    user_id: int,
    user_item: pd.DataFrame,
    item_sim: np.ndarray,
    item_ids: np.ndarray,
    top_n: int = 10,
    exclude_rated: bool = True
) -> List[Tuple[int, float]]:
    if user_id not in user_item.index:
        return []
    user_vec = user_item.loc[user_id].to_numpy()               # ratings for items
    scores = item_sim @ user_vec                                # aggregate similarity
    if exclude_rated:
        scores[user_vec > 0] = -np.inf                          # mask already rated
    top_idx = np.argsort(-scores)[:top_n]
    return [(int(item_ids[i]), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]
