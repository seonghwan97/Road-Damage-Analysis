from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    image_id: str
    captured_at: datetime | None = None
    latitude: float
    longitude: float
    image_url: str
    local_path: Path | None = None
    pano_id: str | None = None
    heading: float | None = None
    pitch: float | None = None
    source: str = "google_street_view"


class DamageDetection(BaseModel):
    image_id: str
    pano_id: str | None = None
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_area_ratio: float = Field(ge=0.0, le=1.0)
    damage_score: float
    latitude: float
    longitude: float
    image_width: int | None = None
    image_height: int | None = None
    local_path: Path | None = None
    heading: float | None = None
    pitch: float | None = None
    captured_at: datetime | None = None
    source: str = "google_street_view"
