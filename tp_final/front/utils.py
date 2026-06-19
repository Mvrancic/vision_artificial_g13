from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


TP_FINAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TP_FINAL_DIR.parent
STATIC_IMAGES_DIR = REPO_ROOT / "static" / "images"

PERSPECTIVE_SAMPLE = STATIC_IMAGES_DIR / "pool2.png"
RENOVATION_SAMPLE = STATIC_IMAGES_DIR / "messi.jpg"
RECONSTRUCTION_SAMPLE = STATIC_IMAGES_DIR / "road.jpg"


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def load_rgb_image(path: str | os.PathLike[str]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    return bgr_to_rgb(image)


def resize_for_display(image: np.ndarray, max_width: int = 960, max_height: int = 720) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_points(image: np.ndarray, points: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    canvas = image.copy()
    for idx, point in enumerate(points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), 7, (255, 64, 64), -1)
        cv2.putText(
            canvas,
            str(idx),
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    if len(points) >= 2:
        poly = np.array(points, dtype=np.int32)
        cv2.polylines(canvas, [poly], len(points) == 4, (64, 255, 128), 2)
    return canvas


def draw_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    canvas = image.copy()
    if bbox is None:
        return canvas
    x, y, w, h = bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 64, 64), 3)
    cv2.putText(
        canvas,
        f"ROI {w}x{h}",
        (x + 4, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return canvas


def build_bbox_from_points(points: list[list[int]] | list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(points) < 2:
        return None
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None
    return x_min, y_min, width, height


def status_for_points(points: list[list[int]] | list[tuple[int, int]], expected: int) -> str:
    count = len(points)
    if count >= expected:
        return f"Listo: {count}/{expected} puntos marcados."
    return f"Marcá {expected} puntos. Estado actual: {count}/{expected}."
