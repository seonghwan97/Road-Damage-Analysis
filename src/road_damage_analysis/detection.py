from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import typer
from PIL import Image, ImageOps

from road_damage_analysis.config import ModelConfig
from road_damage_analysis.schemas import DamageDetection, ImageRecord


class RoadDamageDetector:
    def __init__(self, config: ModelConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.weights_path = project_root / config.weights_path
        self._ensure_weights()
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))
        self.device = _resolve_device(config.device)

    def run(self, images: list[ImageRecord]) -> pd.DataFrame:
        rows: list[dict] = []
        # Run inference image-by-image so progress is visible in the CLI.
        with typer.progressbar(images, label="Running damage detection inference") as progress:
            for image in progress:
                if image.local_path is None:
                    continue
                with Image.open(image.local_path) as pil_image:
                    base_image = pil_image.convert('RGB')
                    width, height = base_image.size
                    # Crop the bottom watermark band before inference.
                    cropped_image, crop_offset_y = self._prepare_image(base_image)
                    detections = self._predict_with_tta(cropped_image, crop_offset_y)
                fused = self._fuse_detections(detections, width, height)
                for detection_row in fused:
                    class_name = detection_row['class_name']
                    severity = self.config.class_severity.get(class_name, 0.5)
                    area_ratio = max(((detection_row['x2'] - detection_row['x1']) * (detection_row['y2'] - detection_row['y1'])) / float(width * height), 0.0)
                    damage_score = min(detection_row['confidence'] * severity * (1.0 + area_ratio * 10.0) * 100.0, 100.0)
                    detection = DamageDetection(
                        image_id=image.image_id,
                        pano_id=image.pano_id,
                        class_name=class_name,
                        confidence=detection_row['confidence'],
                        bbox_x1=detection_row['x1'],
                        bbox_y1=detection_row['y1'],
                        bbox_x2=detection_row['x2'],
                        bbox_y2=detection_row['y2'],
                        bbox_area_ratio=area_ratio,
                        damage_score=damage_score,
                        latitude=image.latitude,
                        longitude=image.longitude,
                        image_width=width,
                        image_height=height,
                        local_path=image.local_path,
                        heading=image.heading,
                        pitch=image.pitch,
                        captured_at=image.captured_at,
                        source=image.source,
                    )
                    rows.append(detection.model_dump(mode='json'))
        return pd.DataFrame(rows)

    def _prepare_image(self, image: Image.Image) -> tuple[Image.Image, int]:
        width, height = image.size
        crop_ratio = min(max(float(self.config.crop_bottom_ratio), 0.0), 0.25)
        crop_pixels = int(round(height * crop_ratio))
        if crop_pixels <= 0:
            return image, 0
        cropped_height = max(1, height - crop_pixels)
        return image.crop((0, 0, width, cropped_height)), 0

    def _predict_with_tta(self, image: Image.Image, crop_offset_y: int) -> list[dict]:
        # Use a minimal TTA set to improve stability without doubling complexity.
        transforms = [('original', image)]
        if self.config.enable_tta:
            transforms.append(('hflip', ImageOps.mirror(image)))

        all_detections: list[dict] = []
        width, _ = image.size
        conf_threshold = max(0.5, float(self.config.conf_threshold))

        for transform_name, transformed_image in transforms:
            results = self.model.predict(
                source=transformed_image,
                conf=conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                device=self.device,
                verbose=False,
            )
            for result in results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    if confidence < conf_threshold:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    if transform_name == 'hflip':
                        x1, x2 = width - x2, width - x1
                    y1 += crop_offset_y
                    y2 += crop_offset_y
                    class_name = str(names.get(cls_id, cls_id)).lower()
                    all_detections.append(
                        {
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                        }
                    )
        return all_detections

    def _fuse_detections(self, detections: list[dict], width: int, height: int) -> list[dict]:
        if not detections:
            return []
        threshold = max(0.5, float(self.config.wbf_skip_box_threshold), float(self.config.conf_threshold))
        filtered = [item for item in detections if float(item['confidence']) >= threshold]
        if not self.config.enable_wbf:
            return filtered

        fused: list[dict] = []
        by_class: dict[str, list[dict]] = {}
        for item in filtered:
            by_class.setdefault(item['class_name'], []).append(item)

        # Merge overlapping boxes of the same class after TTA.
        for class_name, class_items in by_class.items():
            remaining = sorted(class_items, key=lambda item: item['confidence'], reverse=True)
            while remaining:
                anchor = remaining.pop(0)
                group = [anchor]
                leftovers: list[dict] = []
                for candidate in remaining:
                    if _iou(anchor, candidate) >= self.config.wbf_iou_threshold:
                        group.append(candidate)
                    else:
                        leftovers.append(candidate)
                remaining = leftovers
                total_weight = sum(item['confidence'] for item in group)
                if total_weight <= 0:
                    continue
                fused.append(
                    {
                        'class_name': class_name,
                        'confidence': sum(item['confidence'] for item in group) / len(group),
                        'x1': sum(item['x1'] * item['confidence'] for item in group) / total_weight,
                        'y1': sum(item['y1'] * item['confidence'] for item in group) / total_weight,
                        'x2': sum(item['x2'] * item['confidence'] for item in group) / total_weight,
                        'y2': sum(item['y2'] * item['confidence'] for item in group) / total_weight,
                    }
                )
        return [item for item in fused if item['confidence'] >= threshold and _valid_box(item, width, height)]

    def _ensure_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        if self.weights_path.exists():
            return
        weights_url = (self.config.weights_url or '').strip()
        if not weights_url or _looks_like_placeholder_url(weights_url):
            raise FileNotFoundError(
                'Missing road-damage model weights. Place a pretrained .pt file at '
                f'{self.weights_path} or set MODEL_WEIGHTS_URL in .env to a real downloadable weights URL.'
            )
        response = requests.get(weights_url, timeout=120)
        response.raise_for_status()
        self.weights_path.write_bytes(response.content)


def _valid_box(box: dict, width: int, height: int) -> bool:
    return 0 <= box['x1'] < box['x2'] <= width and 0 <= box['y1'] < box['y2'] <= height


def _iou(box_a: dict, box_b: dict) -> float:
    inter_x1 = max(box_a['x1'], box_b['x1'])
    inter_y1 = max(box_a['y1'], box_b['y1'])
    inter_x2 = min(box_a['x2'], box_b['x2'])
    inter_y2 = min(box_a['y2'], box_b['y2'])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, box_a['x2'] - box_a['x1']) * max(0.0, box_a['y2'] - box_a['y1'])
    area_b = max(0.0, box_b['x2'] - box_b['x1']) * max(0.0, box_b['y2'] - box_b['y1'])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _looks_like_placeholder_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.netloc or '').lower()
    return (
        not parsed.scheme
        or not host
        or 'your-hosted-road-damage-model' in value.lower()
        or 'replace_with' in value.lower()
    )


def _resolve_device(configured: str) -> str:
    target = (configured or 'auto').strip().lower()
    if target not in {'auto', 'cpu'}:
        return configured
    if target == 'cpu':
        return 'cpu'
    try:
        import torch

        return '0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'
