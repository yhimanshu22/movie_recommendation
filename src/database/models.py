# Simple models (data contracts) for DB layer

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str

@dataclass
class Rating:
    user_id: int
    movie_id: int
    rating: float

@dataclass
class WatchItem:
    user_id: int
    movie_id: int
