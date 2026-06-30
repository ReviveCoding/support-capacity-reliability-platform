from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

NYC_311_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"


def nyc311_query_url(limit: int = 50_000, start_date: str | None = None) -> str:
    params: dict[str, str | int] = {
        "$limit": limit,
        "$order": "created_date ASC",
        "$select": "unique_key,created_date,closed_date,agency,complaint_type,descriptor,status,borough,incident_zip",
    }
    if start_date:
        params["$where"] = f"created_date >= '{start_date}'"
    return f"{NYC_311_ENDPOINT}?{urlencode(params)}"


def load_nyc311_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"created_date", "complaint_type", "agency"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NYC 311 file is missing columns: {sorted(missing)}")
    frame["created_date"] = pd.to_datetime(frame["created_date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created_date", "complaint_type"])
    frame["region"] = frame.get("borough", "UNKNOWN").fillna("UNKNOWN").astype(str)
    frame["skill"] = frame["complaint_type"].astype(str)
    frame["timestamp"] = frame["created_date"].dt.floor("30min")
    aggregated = (
        frame.groupby(["timestamp", "region", "skill"], as_index=False)
        .size()
        .rename(columns={"size": "offered_contacts"})
    )
    aggregated["source_type"] = "public_observed_proxy"
    return aggregated
