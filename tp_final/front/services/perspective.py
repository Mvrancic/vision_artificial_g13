from __future__ import annotations

import cv2
import numpy as np

from tp_final.common import assets, geometry
from tp_final.front import utils


FURNITURE_ITEMS = assets.load_furniture()
FURNITURE_MAP = {name: sprite for name, sprite in FURNITURE_ITEMS}


def furniture_names() -> list[str]:
    return list(FURNITURE_MAP.keys())


def annotate_selection(image: np.ndarray, points: list[list[int]]) -> np.ndarray:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")
    return utils.draw_points(rgb, points)


def _rgba_alpha_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3]
    return None


def _extract_editor_points(editor_value) -> tuple[np.ndarray, list[list[int]]]:
    if not isinstance(editor_value, dict):
        raise ValueError("No hay imagen cargada.")

    background = utils.ensure_rgb(editor_value.get("background"))
    if background is None:
        raise ValueError("No hay imagen cargada.")

    mask = np.zeros(background.shape[:2], dtype=np.uint8)
    for layer in editor_value.get("layers") or []:
        alpha = _rgba_alpha_mask(layer)
        if alpha is not None:
            mask = np.maximum(mask, alpha.astype(np.uint8))

    if not np.any(mask):
        raise ValueError("Marcá 4 puntos sobre el piso antes de continuar.")

    _, binary = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components: list[tuple[int, list[int]]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        cx, cy = centroids[label]
        components.append((area, [int(round(cx)), int(round(cy))]))

    if len(components) < 4:
        raise ValueError("Necesitás marcar 4 puntos bien separados sobre el piso.")

    components.sort(key=lambda item: item[0], reverse=True)
    points = [point for _, point in components[:4]]
    return background, points


def annotate_editor_selection(editor_value) -> tuple[np.ndarray, str]:
    background, points = _extract_editor_points(editor_value)
    if len(points) != 4:
        raise ValueError("Necesitás marcar exactamente 4 puntos.")
    ordered = geometry.order_points(np.array(points, dtype="float32")).astype(int).tolist()
    preview = utils.draw_points(background, ordered)
    return preview, "Piso previsualizado. Si quedó mal, limpiá las marcas y volvé a pintar 4 puntos."


def render_result(
    image: np.ndarray,
    points: list[list[int]],
    furniture_name: str,
    height_ratio: float,
    shadow_opacity: float,
    show_shadow: bool,
) -> np.ndarray:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")
    if len(points) != 4:
        raise ValueError("Necesitás marcar exactamente 4 puntos del piso.")
    if furniture_name not in FURNITURE_MAP:
        raise ValueError(f"Mueble desconocido: {furniture_name}")

    background = utils.rgb_to_bgr(rgb)
    floor_quad = geometry.order_points(np.array(points, dtype="float32"))
    sprite = FURNITURE_MAP[furniture_name]

    out = background.copy()
    if show_shadow and shadow_opacity > 0:
        light = geometry.estimate_light_direction(background)
        out = geometry.project_shadow(out, floor_quad, light, shadow_opacity)

    standing = geometry.standing_quad_from_floor(floor_quad, height_ratio)
    bgr, alpha = geometry.warp_sprite_to_quad(sprite, standing, out.shape[:2])
    out = geometry.alpha_composite(out, bgr, alpha)
    return utils.bgr_to_rgb(out)


def render_from_editor(
    editor_value,
    furniture_name: str,
    height_ratio: float,
    shadow_opacity: float,
    show_shadow: bool,
) -> np.ndarray:
    background, points = _extract_editor_points(editor_value)
    ordered = geometry.order_points(np.array(points, dtype="float32")).astype(int).tolist()
    return render_result(
        background,
        ordered,
        furniture_name,
        height_ratio,
        shadow_opacity,
        show_shadow,
    )
