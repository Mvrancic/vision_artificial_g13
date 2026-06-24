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


def load_custom_sprite(filepath: str) -> tuple[np.ndarray, str | None]:
    """Carga un PNG subido por el usuario como sprite RGBA (BGR + alfa).

    Si la imagen no tiene canal alfa (no es un PNG con fondo transparente), se
    la deja igual pero se devuelve una advertencia: sin transparencia el recorte
    se va a ver como un rectangulo opaco, no como el mueble recortado.
    """
    sprite = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if sprite is None:
        raise ValueError(f"No se pudo leer la imagen: {filepath}")

    warning = None
    if sprite.ndim == 2:
        sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)
        warning = "La imagen no tiene canal alfa: se va a ver como un rectángulo opaco. Usá un PNG con fondo transparente."
    elif sprite.shape[2] == 3:
        alpha = np.full(sprite.shape[:2], 255, dtype=np.uint8)
        sprite = np.dstack([sprite, alpha])
        warning = "La imagen no tiene canal alfa: se va a ver como un rectángulo opaco. Usá un PNG con fondo transparente."

    return sprite, warning


def floor_contact_point(
    points: list[list[int]],
    height_ratio: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_deg: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[int, int]:
    """Punto (u, v) en pixeles de la imagen donde el mueble toca el piso.

    Es el punto medio de la arista de base del cuadrilatero parado, despues de
    aplicar el mismo mover/rotar/escalar que se uso para renderizarlo. Se usa
    para ubicar el mueble en el plano de Reconstruccion sin depender de que
    la red de profundidad lo "redescubra" en la foto (no tiene forma de saberlo,
    es un sprite pegado, no un objeto real fotografiado).
    """
    floor_quad = geometry.order_points(np.array(points, dtype="float32"))
    standing = geometry.standing_quad_from_floor(floor_quad, height_ratio)
    standing = geometry.transform_quad(
        standing,
        offset=(float(offset_x), float(offset_y)),
        rotation_deg=rotation_deg,
        scale=(float(scale_x), float(scale_y)),
    )
    base_right, base_left = standing[2], standing[3]
    contact = (base_right + base_left) / 2.0
    return int(round(contact[0])), int(round(contact[1]))


def render_result(
    image: np.ndarray,
    points: list[list[int]],
    furniture_name: str,
    height_ratio: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_deg: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    custom_sprite: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")
    if len(points) != 4:
        raise ValueError("Marcá 4 puntos sobre el piso antes de renderizar.")

    if custom_sprite is not None:
        sprite = custom_sprite
    else:
        if furniture_name not in FURNITURE_MAP:
            raise ValueError(f"Mueble desconocido: {furniture_name}")
        sprite = FURNITURE_MAP[furniture_name]

    background = utils.rgb_to_bgr(rgb)
    floor_quad = geometry.order_points(np.array(points, dtype="float32"))
    offset = (float(offset_x), float(offset_y))

    standing = geometry.standing_quad_from_floor(floor_quad, height_ratio)
    standing = geometry.transform_quad(
        standing, offset=offset, rotation_deg=rotation_deg, scale=(float(scale_x), float(scale_y))
    )
    bgr, alpha = geometry.warp_sprite_to_quad(sprite, standing, background.shape[:2])
    out = geometry.alpha_composite(background, bgr, alpha)

    base_right, base_left = standing[2], standing[3]
    contact = (base_right + base_left) / 2.0
    contact_point = (int(round(contact[0])), int(round(contact[1])))
    return utils.bgr_to_rgb(out), contact_point
