# Basic explanation utilities for recommendations

from __future__ import annotations
import re
from typing import List

def top_overlap_terms(seed_tags: str, rec_tags: str, top_k: int = 5) -> List[str]:
    s = set(re.findall(r"[a-zA-Z]+", seed_tags.lower()))
    r = set(re.findall(r"[a-zA-Z]+", rec_tags.lower()))
    terms = list(s & r)
    return terms[:top_k]

def build_explanation(seed_title: str, rec_title: str, terms: List[str]) -> str:
    if not terms:
        return f"Because you watched **{seed_title}**."
    return f"Because you watched **{seed_title}** — shared themes: *{', '.join(terms)}*."
