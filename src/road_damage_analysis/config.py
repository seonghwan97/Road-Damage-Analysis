from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    west: float
    south: float
    east: float
    north: float


class ProjectConfig(BaseModel):
    area_name: str = "Custom Area"
    output_dir: Path = Path("data/processed")
    raw_dir: Path = Path("data/raw")


class ImageryConfig(BaseModel):
    source: str = "google_street_view"
    bbox: BoundingBox
    captured_after: str
    sample_spacing_deg: float = 0.002
    road_sample_spacing_m: float = 35.0
    max_locations: int = 400
    headings: list[int] = Field(default_factory=lambda: [0, 90, 180, 270])
    pitch: int = 0
    fov: int = 90
    radius_m: int = 50
    image_width: int = 640
    image_height: int = 640
    source_preference: str = "outdoor"
    prefer_road_heading: bool = True
    bidirectional_headings: bool = True
    heading_bin_deg: int = 30
    pano_snap_max_distance_m: float = 12.0


class ModelConfig(BaseModel):
    framework: str = "ultralytics"
    weights_path: Path = Path("models/road_damage_best.pt")
    weights_url: str = ""
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    image_size: int = 1280
    device: str = "cpu"
    enable_tta: bool = True
    enable_wbf: bool = True
    wbf_iou_threshold: float = 0.55
    wbf_skip_box_threshold: float = 0.5
    crop_bottom_ratio: float = 0.06
    class_severity: dict[str, float] = Field(default_factory=dict)


class AggregationConfig(BaseModel):
    grid_cell_size_deg: float = 0.01
    nearest_segment_radius_m: float = 40.0
    heatmap_radius: int = 24


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    tiles_url: str
    map_center: dict[str, float] | None = None
    initial_zoom: int = 10


class Settings(BaseModel):
    project: ProjectConfig
    imagery: ImageryConfig
    model: ModelConfig
    aggregation: AggregationConfig
    web: WebConfig

    def ensure_directories(self, root: Path) -> None:
        for relative in [
            self.project.output_dir,
            self.project.raw_dir,
            Path("models"),
            Path("web"),
        ]:
            (root / relative).mkdir(parents=True, exist_ok=True)


def load_settings(config_path: Path) -> Settings:
    with config_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file)
    return Settings.model_validate(data)
