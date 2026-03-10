from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from road_damage_analysis.config import Settings


def _sanitize_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    return _sanitize_json(json.loads(path.read_text(encoding='utf-8')))


def create_app(project_root: Path, settings: Settings) -> FastAPI:
    output_dir = project_root / settings.project.output_dir
    web_dir = project_root / 'web'
    raw_dir = project_root / settings.project.raw_dir
    app = FastAPI(title='Global Road Damage Monitor')
    map_center = settings.web.map_center or {
        'lat': (settings.imagery.bbox.south + settings.imagery.bbox.north) / 2.0,
        'lon': (settings.imagery.bbox.west + settings.imagery.bbox.east) / 2.0,
    }
    app.mount('/media', StaticFiles(directory=raw_dir), name='media')

    @app.get('/', response_class=HTMLResponse)
    def index() -> str:
        template = (web_dir / 'index.html').read_text(encoding='utf-8')
        return (
            template.replace('__MAP_CENTER_LAT__', str(map_center['lat']))
            .replace('__MAP_CENTER_LON__', str(map_center['lon']))
            .replace('__MAP_ZOOM__', str(settings.web.initial_zoom))
            .replace('__TILES_URL__', settings.web.tiles_url)
            .replace('__HEAT_RADIUS__', str(settings.aggregation.heatmap_radius))
            .replace('__AREA_NAME__', settings.project.area_name)
        )

    @app.get('/api/summary')
    def summary() -> JSONResponse:
        path = output_dir / 'summary.json'
        payload = _load_json(path, {})
        return JSONResponse(payload)

    @app.get('/api/detections.geojson')
    def detections() -> JSONResponse:
        path = output_dir / 'damage_detections.geojson'
        payload = _load_json(path, {'type': 'FeatureCollection', 'features': []})
        return JSONResponse(payload)

    @app.get('/api/grid.geojson')
    def grid() -> JSONResponse:
        path = output_dir / 'damage_grid.geojson'
        payload = _load_json(path, {'type': 'FeatureCollection', 'features': []})
        return JSONResponse(payload)

    @app.get('/api/segments.geojson')
    def segments() -> JSONResponse:
        path = output_dir / 'damage_segments.geojson'
        payload = _load_json(path, {'type': 'FeatureCollection', 'features': []})
        return JSONResponse(payload)

    return app
