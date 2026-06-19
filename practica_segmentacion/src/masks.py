from __future__ import annotations

import cv2
import numpy as np


def keep_largest_components(mask: np.ndarray, max_components: int = 3) -> np.ndarray:
    """Keep the largest foreground components and remove small artifacts."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    ordered = np.argsort(areas)[::-1][:max_components] + 1
    clean = np.isin(labels, ordered).astype(np.uint8) * 255
    return clean


def generate_cell_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Generate a binary pseudo-mask for the visible cell region."""
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    _, sat_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    chroma = cv2.absdiff(a_channel, np.full_like(a_channel, 128))
    chroma = cv2.add(chroma, cv2.absdiff(b_channel, np.full_like(b_channel, 128)))
    _, chroma_mask = cv2.threshold(
        chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_or(sat_mask, chroma_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = keep_largest_components(mask, max_components=4)
    mask = cv2.medianBlur(mask, 5)

    return (mask > 0).astype(np.uint8) * 255


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = np.zeros_like(image_bgr)
    color[:, :, 2] = 255
    blended = cv2.addWeighted(image_bgr, 1.0, color, alpha, 0)
    return np.where(mask[:, :, None] > 0, blended, image_bgr)
