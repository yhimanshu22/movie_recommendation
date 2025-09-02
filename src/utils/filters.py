# Simple filters for search/results

from __future__ import annotations
import pandas as pd
from typing import Optional, Tuple

def apply_filters(
    df: pd.DataFrame,
    year_range: Optional[Tuple[int,int]] = None,
    language: Optional[str] = None,
    min_vote: Optional[float] = None
) -> pd.DataFrame:
    out = df.copy()
    if "release_date" in out.columns and year_range:
        out["year"] = pd.to_datetime(out["release_date"], errors="coerce").dt.year
        lo, hi = year_range
        out = out[(out["year"] >= lo) & (out["year"] <= hi)]
    if language and "original_language" in out.columns:
        out = out[out["original_language"].str.lower() == language.lower()]
    if min_vote is not None and "vote_average" in out.columns:
        out = out[out["vote_average"] >= float(min_vote)]
    return out
