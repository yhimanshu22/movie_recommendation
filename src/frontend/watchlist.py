# Watchlist helpers using session_state; can be backed by DB later.

from __future__ import annotations
import streamlit as st
from typing import List

def init_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

def add_to_watchlist(movie_id: int):
    init_watchlist()
    if movie_id not in st.session_state.watchlist:
        st.session_state.watchlist.append(movie_id)

def remove_from_watchlist(movie_id: int):
    init_watchlist()
    st.session_state.watchlist = [m for m in st.session_state.watchlist if m != movie_id]

def get_watchlist() -> List[int]:
    init_watchlist()
    return st.session_state.watchlist
