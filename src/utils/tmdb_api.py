# TMDB helper: posters & trailers. Requires env TMDB_API_KEY.

from __future__ import annotations
import os, requests
from typing import Optional

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"

def _get(path: str, params: dict) -> dict:
    if not TMDB_API_KEY:
        return {}
    url = f"{TMDB_BASE}/{path}"
    params = {"api_key": TMDB_API_KEY, **params}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return {}
    return r.json()

def search_movie(title: str) -> Optional[dict]:
    data = _get("search/movie", {"query": title})
    if not data or not data.get("results"):
        return None
    return data["results"][0]

def get_poster_url(poster_path: str | None, size: str = "w342") -> Optional[str]:
    if not poster_path:
        return None
    return f"https://image.tmdb.org/t/p/{size}{poster_path}"

def get_trailer_youtube_key(movie_id: int) -> Optional[str]:
    data = _get(f"movie/{movie_id}/videos", {})
    if not data or not data.get("results"):
        return None
    for v in data["results"]:
        if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
            return v.get("key")
    return None
