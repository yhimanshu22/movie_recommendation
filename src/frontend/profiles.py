# Dummy profiles; replace with DB-backed profiles later.

from __future__ import annotations
import streamlit as st

DEFAULT_PROFILES = ["Guest", "ActionFan", "RomComLover", "SciFiNerd"]

def select_profile() -> str:
    if "profile" not in st.session_state:
        st.session_state.profile = DEFAULT_PROFILES[0]
    st.session_state.profile = st.sidebar.selectbox("Profile", DEFAULT_PROFILES, index=0)
    return st.session_state.profile
