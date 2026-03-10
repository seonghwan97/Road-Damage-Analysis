from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
import typer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from road_damage_analysis.config import BoundingBox, ImageryConfig
from road_damage_analysis.schemas import ImageRecord


METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass
class RoadGeometry:
    road_id: str
    line: LineString
    road_name: str | None = None
    highway: str | None = None


@dataclass
class RoadCandidate:
    latitude: float
    longitude: float
    bearing: float
    road_id: str


@dataclass
class RoadBearingIndex:
    tree: STRtree
    roads: list[RoadGeometry]
    road_map: dict[str, RoadGeometry]


class GoogleStreetViewCollector:
    def __init__(self, api_key: str, config: ImageryConfig, raw_dir: Path, area_name: str) -> None:
        self.api_key = api_key
        self.config = config
        self.raw_dir = raw_dir
        self.area_name = area_name
        self.session = requests.Session()
        self.road_index = self._build_road_index() if config.prefer_road_heading else None

    def collect_metadata(self) -> list[ImageRecord]:
        # Collect one metadata record per pano/heading bucket combination.
        records: list[ImageRecord] = []
        seen_image_keys: set[tuple[str, int]] = set()
        captured_after = _parse_iso_date(self.config.captured_after)

        for candidate in self._candidate_points():
            if len(records) >= self.config.max_locations:
                break
            # Resolve the nearest pano first, then derive headings from the matched road.
            metadata = self._fetch_metadata(candidate.latitude, candidate.longitude)
            if not metadata or metadata.get("status") != "OK":
                continue
            captured_at = _parse_street_view_date(metadata.get("date"))
            if captured_at and captured_at < captured_after:
                continue
            pano_id = str(metadata.get("pano_id", "")).strip()
            if not pano_id:
                continue
            location = metadata.get("location") or {}
            latitude = float(location.get("lat", candidate.latitude))
            longitude = float(location.get("lng", candidate.longitude))
            headings = self._resolve_headings(candidate, latitude, longitude)
            for heading in headings:
                dedupe_key = (pano_id, _heading_bucket(heading, self.config.heading_bin_deg))
                if dedupe_key in seen_image_keys:
                    continue
                image_url = self._build_image_url(latitude, longitude, heading, pano_id)
                image_id = f"{pano_id}_{dedupe_key[1]}_{self.config.pitch}"
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        pano_id=pano_id,
                        captured_at=captured_at,
                        latitude=latitude,
                        longitude=longitude,
                        image_url=image_url,
                        heading=float(heading),
                        pitch=float(self.config.pitch),
                        source="google_street_view",
                    )
                )
                seen_image_keys.add(dedupe_key)
                if len(records) >= self.config.max_locations:
                    break
        return records

    def download_images(self, records: list[ImageRecord], limit: int | None = None) -> list[ImageRecord]:
        output: list[ImageRecord] = []
        target_dir = self.raw_dir / _slugify(self.area_name) / "google_street_view_images"
        target_dir.mkdir(parents=True, exist_ok=True)
        download_records = records[:limit]
        # Show download progress because image collection can take a while.
        with typer.progressbar(download_records, label="Downloading Street View images") as progress:
            for record in progress:
                path = target_dir / f"{record.image_id}.jpg"
                if not path.exists():
                    response = self.session.get(record.image_url, timeout=60)
                    response.raise_for_status()
                    path.write_bytes(response.content)
                output.append(record.model_copy(update={"local_path": path}))
        return output

    def _candidate_points(self) -> list[RoadCandidate]:
        if self.road_index is not None:
            sampled = _sample_roads(self.road_index.roads, self.config.road_sample_spacing_m)
            if sampled:
                return sampled
        return [
            RoadCandidate(latitude=lat, longitude=lon, bearing=float(self.config.headings[0]), road_id="grid")
            for lat, lon in _iter_grid_points(self.config.bbox, self.config.sample_spacing_deg)
        ]

    def _resolve_headings(self, candidate: RoadCandidate, latitude: float, longitude: float) -> list[int]:
        if self.road_index is None or candidate.road_id == 'grid':
            return self._fallback_headings(candidate)
        road = self.road_index.road_map.get(candidate.road_id)
        if road is None:
            return self._fallback_headings(candidate)
        point = Point(longitude, latitude)
        distance_m = _distance_point_to_line_m(point, road.line)
        # Reject panos that snap onto a different nearby road segment.
        if distance_m > self.config.pano_snap_max_distance_m:
            return []
        bearing = _local_bearing_on_line(road.line, point)
        if bearing is None:
            bearing = candidate.bearing
        headings = [round(bearing) % 360]
        if self.config.bidirectional_headings:
            headings.append((headings[0] + 180) % 360)
        return list(dict.fromkeys(headings))

    def _fallback_headings(self, candidate: RoadCandidate) -> list[int]:
        if self.road_index is None:
            return [int(h) % 360 for h in self.config.headings]
        headings = [round(candidate.bearing) % 360]
        if self.config.bidirectional_headings:
            headings.append((headings[0] + 180) % 360)
        return list(dict.fromkeys(headings))

    def _fetch_metadata(self, latitude: float, longitude: float) -> dict | None:
        params = {
            "location": f"{latitude},{longitude}",
            "radius": self.config.radius_m,
            "source": self.config.source_preference,
            "key": self.api_key,
        }
        try:
            response = self.session.get(METADATA_URL, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def _build_image_url(self, latitude: float, longitude: float, heading: int, pano_id: str) -> str:
        params = {
            "size": f"{self.config.image_width}x{self.config.image_height}",
            "location": f"{latitude},{longitude}",
            "pano": pano_id,
            "heading": heading,
            "pitch": self.config.pitch,
            "fov": self.config.fov,
            "radius": self.config.radius_m,
            "source": self.config.source_preference,
            "key": self.api_key,
        }
        return f"{IMAGE_URL}?{urlencode(params)}"

    def _build_road_index(self) -> RoadBearingIndex | None:
        query = _overpass_query(self.config.bbox)
        try:
            response = self.session.get(OVERPASS_URL, params={"data": query}, timeout=120)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        nodes = {
            element["id"]: (element["lon"], element["lat"])
            for element in payload.get("elements", [])
            if element.get("type") == "node"
        }
        roads: list[RoadGeometry] = []
        for element in payload.get("elements", []):
            if element.get("type") != "way":
                continue
            refs = element.get("nodes", [])
            coords = [nodes[node_id] for node_id in refs if node_id in nodes]
            if len(coords) < 2:
                continue
            line = LineString(coords)
            tags = element.get("tags", {})
            roads.append(
                RoadGeometry(
                    road_id=str(element.get("id")),
                    line=line,
                    road_name=tags.get("name"),
                    highway=tags.get("highway"),
                )
            )
        if not roads:
            return None
        road_map = {road.road_id: road for road in roads}
        return RoadBearingIndex(tree=STRtree([road.line for road in roads]), roads=roads, road_map=road_map)


def _overpass_query(bbox: BoundingBox) -> str:
    return (
        f"[out:json][timeout:60];"
        f"(way[highway][highway!~\"footway|path|cycleway|steps|pedestrian|track|service|living_street\"]"
        f"({bbox.south},{bbox.west},{bbox.north},{bbox.east}););"
        f"(._;>;);out body;"
    )


def _sample_roads(roads: list[RoadGeometry], spacing_m: float) -> list[RoadCandidate]:
    candidates: list[RoadCandidate] = []
    seen: set[tuple[int, int, int]] = set()
    spacing_m = max(10.0, spacing_m)
    for road in roads:
        coords = list(road.line.coords)
        if len(coords) < 2:
            continue
        carried = 0.0
        previous = coords[0]
        for current in coords[1:]:
            start_lat, start_lon = previous[1], previous[0]
            end_lat, end_lon = current[1], current[0]
            segment_length = _haversine_m(start_lat, start_lon, end_lat, end_lon)
            if segment_length < 3.0:
                previous = current
                continue
            bearing = _bearing(start_lat, start_lon, end_lat, end_lon)
            distance_along = spacing_m - carried if carried > 0 else 0.0
            while distance_along <= segment_length:
                ratio = distance_along / segment_length if segment_length else 0.0
                lat = start_lat + (end_lat - start_lat) * ratio
                lon = start_lon + (end_lon - start_lon) * ratio
                key = (int(round(lat * 1_000_000)), int(round(lon * 1_000_000)), _heading_bucket(bearing, 20))
                if key not in seen:
                    candidates.append(RoadCandidate(latitude=lat, longitude=lon, bearing=bearing, road_id=road.road_id))
                    seen.add(key)
                distance_along += spacing_m
            carried = 0.0 if segment_length == 0 else max(0.0, distance_along - segment_length)
            previous = current
    return candidates


def _iter_grid_points(bbox: BoundingBox, step: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    lat = bbox.south
    while lat <= bbox.north:
        lon = bbox.west
        while lon <= bbox.east:
            points.append((round(lat, 7), round(lon, 7)))
            lon += step
        lat += step
    return points


def _distance_point_to_line_m(point: Point, line: LineString) -> float:
    nearest = line.interpolate(line.project(point))
    return _haversine_m(point.y, point.x, nearest.y, nearest.x)


def _local_bearing_on_line(line: LineString, point: Point) -> float | None:
    coords = list(line.coords)
    best_distance = None
    best_bearing = None
    for start, end in zip(coords, coords[1:]):
        segment = LineString([start, end])
        distance = segment.distance(point)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_bearing = _bearing(start[1], start[0], end[1], end[0])
    return best_bearing


def _parse_iso_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_street_view_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parts = value.split("-")
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        return datetime(year, month, 1, tzinfo=timezone.utc)
    if len(parts) == 1:
        return datetime(int(parts[0]), 1, 1, tzinfo=timezone.utc)
    return None


def _heading_bucket(heading: float, bucket_deg: int) -> int:
    bucket_deg = max(1, int(bucket_deg))
    return (int(round(heading / bucket_deg)) * bucket_deg) % 360


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _slugify(value: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in value.strip())
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'default_area'
