from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from road_damage_analysis.aggregation import attach_nearest_segment, build_grid_summary, build_road_segment_index, segment_geojson, summarize_by_segment
from road_damage_analysis.collectors.google_street_view import GoogleStreetViewCollector
from road_damage_analysis.config import Settings
from road_damage_analysis.detection import RoadDamageDetector
from road_damage_analysis.geo import detections_geojson, grid_geojson
from road_damage_analysis.io_utils import write_dataframe_csv, write_json


def run_pipeline(project_root: Path, settings: Settings, download_limit: int | None = None) -> dict[str, Path]:
    load_dotenv(project_root / '.env')
    settings.ensure_directories(project_root)

    api_key = os.getenv('GOOGLE_STREET_VIEW_API_KEY', '')
    if not api_key:
        raise RuntimeError('GOOGLE_STREET_VIEW_API_KEY is required in .env for Google Street View collection.')

    settings.model.weights_url = os.getenv('MODEL_WEIGHTS_URL', '').strip()
    weights_path_env = os.getenv('MODEL_WEIGHTS_PATH', '').strip()
    if weights_path_env:
        settings.model.weights_path = Path(weights_path_env)

    collector = GoogleStreetViewCollector(api_key, settings.imagery, project_root / settings.project.raw_dir, settings.project.area_name)
    metadata = collector.collect_metadata()
    downloaded = collector.download_images(metadata, limit=download_limit or len(metadata))

    detector = RoadDamageDetector(settings.model, project_root)
    detections = detector.run(downloaded)
    road_index = build_road_segment_index(settings.imagery.bbox)
    detections = attach_nearest_segment(detections, road_index, settings.aggregation)
    grid_summary = build_grid_summary(detections, settings.aggregation)
    segment_summary = summarize_by_segment(detections)

    output_dir = project_root / settings.project.output_dir
    detections_csv = output_dir / 'damage_detections.csv'
    grid_csv = output_dir / 'damage_grid_summary.csv'
    segment_csv = output_dir / 'damage_segment_summary.csv'
    detections_geojson_path = output_dir / 'damage_detections.geojson'
    grid_geojson_path = output_dir / 'damage_grid.geojson'
    segment_geojson_path = output_dir / 'damage_segments.geojson'
    summary_json = output_dir / 'summary.json'

    write_dataframe_csv(detections_csv, detections)
    write_dataframe_csv(grid_csv, grid_summary)
    write_dataframe_csv(segment_csv, segment_summary)
    write_json(detections_geojson_path, detections_geojson(detections, project_root / settings.project.raw_dir))
    write_json(grid_geojson_path, grid_geojson(grid_summary))
    write_json(segment_geojson_path, segment_geojson(road_index, segment_summary))

    summary_payload = {
        'area_name': settings.project.area_name,
        'candidate_locations': len(metadata),
        'images_collected': len(downloaded),
        'detections_count': int(len(detections)),
        'avg_damage_score': float(detections['damage_score'].mean()) if not detections.empty else 0.0,
        'grid_cells': int(len(grid_summary)),
        'segments_flagged': int(len(segment_summary)),
    }
    write_json(summary_json, summary_payload)

    return {
        'detections_csv': detections_csv,
        'grid_csv': grid_csv,
        'segment_csv': segment_csv,
        'detections_geojson': detections_geojson_path,
        'grid_geojson': grid_geojson_path,
        'segment_geojson': segment_geojson_path,
        'summary_json': summary_json,
    }


def load_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        'detections': pd.read_csv(output_dir / 'damage_detections.csv') if (output_dir / 'damage_detections.csv').exists() else pd.DataFrame(),
        'grid': pd.read_csv(output_dir / 'damage_grid_summary.csv') if (output_dir / 'damage_grid_summary.csv').exists() else pd.DataFrame(),
        'segments': pd.read_csv(output_dir / 'damage_segment_summary.csv') if (output_dir / 'damage_segment_summary.csv').exists() else pd.DataFrame(),
    }
