from __future__ import annotations

import cv2
import numpy as np

from tp_final.enfoque2_renovacion.backends import backend_available, segment_object
from tp_final.enfoque2_renovacion.main import remove_and_fill
from tp_final.front import utils


METHOD_MAP = {
    "TELEA": 0,
    "Navier-Stokes": 1,
    "LaMa": 2,
}


def annotate_selection(image: np.ndarray, points: list[list[int]]) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")
    bbox = utils.build_bbox_from_points(points)
    preview = utils.draw_points(rgb, points)
    preview = utils.draw_bbox(preview, bbox)
    return preview, bbox


def run_renovation_from_points(
    image: np.ndarray,
    points: list[list[int]],
    radius: int,
    method_name: str,
    dilate_iter: int,
    expand_px: int,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")

    rect = utils.build_bbox_from_points(points)
    if rect is None:
        raise ValueError("Marcá 2 puntos (esquina superior izq. y esquina inferior der.) sobre el objeto a borrar.")

    bgr = utils.rgb_to_bgr(rgb)
    object_mask, segment_name = segment_object(bgr, rect, mode="auto")
    result, dilated_mask, backend_name = remove_and_fill(
        bgr,
        object_mask,
        radius=int(radius),
        method=METHOD_MAP[method_name],
        dilate_iter=int(dilate_iter),
        expand_px=int(expand_px),
        feather_px=int(feather_px),
    )

    masked = bgr.copy()
    masked[dilated_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(bgr, 0.6, masked, 0.4, 0)
    result_rgb = utils.bgr_to_rgb(result)
    cv2.putText(
        result_rgb,
        f"{segment_name} -> {backend_name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return (
        utils.bgr_to_rgb(overlay),
        result_rgb,
        utils.bgr_to_rgb(cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)),
    )


def get_backend_status(method_name: str) -> str:
    available, message = backend_available(method_name)
    if available:
        return "Backend listo."
    return message or "Backend no disponible."
