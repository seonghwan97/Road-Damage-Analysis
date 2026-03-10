from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from road_damage_analysis.config import AggregationConfig, BoundingBox

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def build_grid_summary(detections: pd.DataFrame, config: AggregationConfig) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=["grid_id", "center_lat", "center_lon", "damage_score_mean", "damage_count", "top_class"])
    cell = config.grid_cell_size_deg
    frame = detections.copy()
    frame["grid_lat"] = (frame["latitude"] / cell).round().astype(int) * cell
    frame["grid_lon"] = (frame["longitude"] / cell).round().astype(int) * cell
    grouped = (
        frame.groupby(["grid_lat", "grid_lon"], dropna=False)
        .agg(
            damage_score_mean=("damage_score", "mean"),
            damage_count=("damage_score", "size"),
            top_class=("class_name", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )
    grouped["grid_id"] = grouped.index.astype(str)
    grouped.rename(columns={"grid_lat": "center_lat", "grid_lon": "center_lon"}, inplace=True)
    return grouped


@dataclass
class RoadSegmentIndex:
    tree: STRtree
    geometries: list[LineString]
    properties: list[dict[str, Any]]


def build_road_segment_index(bbox: BoundingBox) -> RoadSegmentIndex | None:
    query = (
        f"[out:json][timeout:60];"
        f"(way[highway][highway!~\"footway|path|cycleway|steps|pedestrian|track|service\"]"
        f"({bbox.south},{bbox.west},{bbox.north},{bbox.east}););"
        f"(._;>;);out body;"
    )
    try:
        response = requests.get(OVERPASS_URL, params={"data": query}, timeout=120)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in payload.get("elements", [])
        if element.get("type") == "node"
    }
    geometries: list[LineString] = []
    properties: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        refs = element.get("nodes", [])
        coords = [nodes[node_id] for node_id in refs if node_id in nodes]
        if len(coords) < 2:
            continue
        for idx, (start, end) in enumerate(zip(coords, coords[1:])):
            if start == end:
                continue
            geometries.append(LineString([start, end]))
            properties.append({
                "segment_id": f"{element['id']}_{idx}",
                "road_name": tags.get("name") or tags.get("ref") or "Unnamed road",
                "highway": tags.get("highway"),
            })
    if not geometries:
        return None
    return RoadSegmentIndex(tree=STRtree(geometries), geometries=geometries, properties=properties)


def attach_nearest_segment(detections: pd.DataFrame, road_index: RoadSegmentIndex | None, config: AggregationConfig) -> pd.DataFrame:
    frame = detections.copy()
    frame["segment_id"] = None
    frame["road_name"] = None
    if detections.empty or road_index is None:
        return frame
    max_distance_deg = config.nearest_segment_radius_m / 111_320.0
    segment_ids: list[str | None] = []
    road_names: list[str | None] = []
    for row in frame.itertuples(index=False):
        point = Point(row.longitude, row.latitude)
        nearest_idx = road_index.tree.nearest(point)
        if nearest_idx is None:
            segment_ids.append(None)
            road_names.append(None)
            continue
        nearest_idx = int(nearest_idx)
        nearest_geom = road_index.geometries[nearest_idx]
        if point.distance(nearest_geom) > max_distance_deg:
            segment_ids.append(None)
            road_names.append(None)
            continue
        props = road_index.properties[nearest_idx]
        segment_ids.append(props["segment_id"])
        road_names.append(props["road_name"])
    frame["segment_id"] = segment_ids
    frame["road_name"] = road_names
    return frame


def summarize_by_segment(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty or "segment_id" not in detections.columns:
        return pd.DataFrame(columns=["segment_id", "road_name", "damage_score_mean", "damage_count", "top_class"])
    frame = detections.dropna(subset=["segment_id"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["segment_id", "road_name", "damage_score_mean", "damage_count", "top_class"])
    return (
        frame.groupby(["segment_id", "road_name"], dropna=False)
        .agg(
            damage_score_mean=("damage_score", "mean"),
            damage_count=("damage_score", "size"),
            top_class=("class_name", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )


def segment_geojson(road_index: RoadSegmentIndex | None, segment_summary: pd.DataFrame) -> dict:
    if road_index is None:
        return {"type": "FeatureCollection", "features": []}
    summary_map = segment_summary.set_index("segment_id").to_dict(orient="index") if not segment_summary.empty else {}
    features = []
    for geometry, props in zip(road_index.geometries, road_index.properties):
        merged = dict(props)
        merged.update(summary_map.get(props["segment_id"], {}))
        coords = [[float(x), float(y)] for x, y in geometry.coords]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": merged,
        })
    return {"type": "FeatureCollection", "features": features}
