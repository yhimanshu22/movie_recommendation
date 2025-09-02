# Hybrid recommender: blend content-based similarity (movies.pkl/similarity.npy)
# with item-based CF scores (from collab_filtering.py)

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from .collab_filtering import fit_item_cf, recommend_for_user

def _title_to_index(df: pd.DataFrame, title: str) -> int | None:
    m = df[df["title"].str.lower() == title.lower()]
    return int(m.index[0]) if not m.empty else None

def content_recommend_from_title(
    df: pd.DataFrame, sim: np.ndarray, title: str, top_n: int = 20
) -> List[Tuple[int, float]]:
    idx = _title_to_index(df, title)
    if idx is None:
        return []
    scores = sim[idx]
    # skip self at idx, take top_n
    order = np.argsort(-scores)
    out = []
    for j in order:
        if j == idx: 
            continue
        out.append((int(df.iloc[j]["id"]), float(scores[j])))
        if len(out) >= top_n:
            break
    return out

def hybrid_recommend(
    user_id: int | None,
    seed_title: str | None,
    df: pd.DataFrame,
    content_sim: np.ndarray,
    alpha: float = 0.6,               # content weight
    top_n: int = 10,
    ratings_path: str = "data/ratings.csv"
) -> List[Tuple[int, float]]:
    # content candidates
    content_candidates = []
    if seed_title:
        content_candidates = content_recommend_from_title(df, content_sim, seed_title, top_n=50)
    content_dict = {mid: s for mid, s in content_candidates}

    # CF candidates
    cf_dict = {}
    try:
        if user_id is not None:
            user_item, item_sim, item_ids = fit_item_cf(ratings_path)
            cf_list = recommend_for_user(user_id, user_item, item_sim, item_ids, top_n=50)
            cf_dict = {mid: s for mid, s in cf_list}
    except Exception:
        # ratings not present or failed -> skip CF
        pass

    # union & blend
    all_ids = set(content_dict) | set(cf_dict)
    scores = []
    for mid in all_ids:
        c = content_dict.get(mid, 0.0)
        u = cf_dict.get(mid, 0.0)
        score = alpha * c + (1 - alpha) * u
        scores.append((mid, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]
