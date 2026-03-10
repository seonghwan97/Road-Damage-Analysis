from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_dataframe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def records_to_geojson(records: Iterable[dict], lon_key: str, lat_key: str) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [record[lon_key], record[lat_key]],
                },
                "properties": {key: value for key, value in record.items() if key not in {lon_key, lat_key}},
            }
            for record in records
        ],
    }


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
