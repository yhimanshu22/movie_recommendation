# Minimal SQLite wrapper

from __future__ import annotations
import sqlite3
from typing import List, Tuple

DB_PATH = "app.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            movie_id INTEGER,
            rating REAL,
            PRIMARY KEY (user_id, movie_id)
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id INTEGER,
            movie_id INTEGER,
            PRIMARY KEY (user_id, movie_id)
        );
        """
    )
    conn.commit()
    conn.close()

def upsert_rating(user_id: int, movie_id: int, rating: float):
    conn = get_conn()
    conn.execute(
        "INSERT INTO ratings(user_id, movie_id, rating) VALUES(?,?,?) "
        "ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating",
        (user_id, movie_id, rating),
    )
    conn.commit(); conn.close()

def get_user_ratings(user_id: int) -> List[Tuple[int, float]]:
    conn = get_conn()
    cur = conn.execute("SELECT movie_id, rating FROM ratings WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def set_watch(user_id: int, movie_id: int, add: bool = True):
    conn = get_conn()
    if add:
        conn.execute("INSERT OR IGNORE INTO watchlist(user_id, movie_id) VALUES(?,?)", (user_id, movie_id))
    else:
        conn.execute("DELETE FROM watchlist WHERE user_id=? AND movie_id=?", (user_id, movie_id))
    conn.commit(); conn.close()

def get_watchlist(user_id: int) -> List[int]:
    conn = get_conn()
    cur = conn.execute("SELECT movie_id FROM watchlist WHERE user_id=?", (user_id,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows
