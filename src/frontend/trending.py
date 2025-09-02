# Simple "Trending" section placeholder (can swap with real metrics later)

from __future__ import annotations
import pandas as pd
from typing import List, Dict

def top_by_popularity(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    col = "popularity" if "popularity" in df.columns else None
    if col:
        return df.sort_values(col, ascending=False).head(k)
    # fallback: by number of similar items (proxy: title length)
    return df.assign(score=df["title"].str.len()).sort_values("score", ascending=False).head(k)

def to_grid_items(df: pd.DataFrame) -> List[Dict]:
    out = []
    for _, r in df.iterrows():
        out.append({"title": r.get("title", ""), "poster_url": None})
    return out
