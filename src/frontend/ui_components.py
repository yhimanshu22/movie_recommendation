# Streamlit UI components (cards, trailer embed, poster grid)

from __future__ import annotations
import streamlit as st
from typing import List, Optional

def poster_card(title: str, genres: str, poster_url: Optional[str]):
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.empty()
        with cols[1]:
            st.markdown(f"### {title}")
            st.caption(genres or "—")

def poster_grid(items: List[dict], cols: int = 5):
    grid = st.columns(cols)
    for i, it in enumerate(items):
        with grid[i % cols]:
            st.image(it.get("poster_url"), caption=it.get("title", ""), use_container_width=True)

def trailer_player(youtube_key: str):
    st.video(f"https://www.youtube.com/watch?v={youtube_key}")
