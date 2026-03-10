from __future__ import annotations

from pathlib import Path

import pandas as pd


CLASS_LABELS = {
    'd00': 'Longitudinal Crack',
    'd10': 'Transverse Crack',
    'd20': 'Alligator Crack',
    'd40': 'Pothole',
}


def detections_geojson(detections: pd.DataFrame, raw_dir: Path) -> dict:
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [row['longitude'], row['latitude']]},
                'properties': {
                    'image_id': row['image_id'],
                    'pano_id': row.get('pano_id'),
                    'class_name': row['class_name'],
                    'class_label': CLASS_LABELS.get(str(row['class_name']).lower(), row['class_name']),
                    'confidence': row['confidence'],
                    'damage_score': row['damage_score'],
                    'heading': row.get('heading'),
                    'captured_at': row.get('captured_at'),
                    'bbox_x1': row.get('bbox_x1'),
                    'bbox_y1': row.get('bbox_y1'),
                    'bbox_x2': row.get('bbox_x2'),
                    'bbox_y2': row.get('bbox_y2'),
                    'image_width': row.get('image_width'),
                    'image_height': row.get('image_height'),
                    'road_name': row.get('road_name'),
                    'segment_id': row.get('segment_id'),
                    'media_url': _media_url(row.get('local_path'), raw_dir),
                },
            }
            for _, row in detections.iterrows()
        ],
    }


def grid_geojson(grid_summary: pd.DataFrame, radius_deg: float = 0.0035) -> dict:
    features = []
    for _, row in grid_summary.iterrows():
        lon = float(row['center_lon'])
        lat = float(row['center_lat'])
        polygon = [
            [lon - radius_deg, lat - radius_deg],
            [lon + radius_deg, lat - radius_deg],
            [lon + radius_deg, lat + radius_deg],
            [lon - radius_deg, lat + radius_deg],
            [lon - radius_deg, lat - radius_deg],
        ]
        features.append(
            {
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [polygon]},
                'properties': {
                    'grid_id': row['grid_id'],
                    'damage_score_mean': row['damage_score_mean'],
                    'damage_count': row['damage_count'],
                    'top_class': row['top_class'],
                    'top_class_label': CLASS_LABELS.get(str(row['top_class']).lower(), row['top_class']),
                },
            }
        )
    return {'type': 'FeatureCollection', 'features': features}


def _media_url(local_path: object, raw_dir: Path) -> str | None:
    if not local_path or pd.isna(local_path):
        return None
    path = Path(str(local_path))
    try:
        rel = path.resolve().relative_to(raw_dir.resolve())
    except Exception:
        return None
    return '/media/' + '/'.join(rel.parts)
