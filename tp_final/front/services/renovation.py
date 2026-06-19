from __future__ import annotations

import cv2
import numpy as np

from tp_final.enfoque2_renovacion.backends import backend_available, clean_mask
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


def _rgba_alpha_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3]
    return None


def _extract_editor_payload(editor_value) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(editor_value, dict):
        raise ValueError("No hay imagen cargada.")

    background = utils.ensure_rgb(editor_value.get("background"))
    if background is None:
        raise ValueError("No hay imagen cargada.")

    mask = np.zeros(background.shape[:2], dtype=np.uint8)
    layers = editor_value.get("layers") or []
    for layer in layers:
        alpha = _rgba_alpha_mask(layer)
        if alpha is not None:
            mask = np.maximum(mask, alpha.astype(np.uint8))

    composite = editor_value.get("composite")
    if composite is not None and not np.any(mask):
        composite_rgb = utils.ensure_rgb(composite)
        if composite_rgb is not None:
            diff = cv2.absdiff(background, composite_rgb)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            mask = np.where(diff_gray > 20, 255, 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("Pintá la zona a eliminar sobre la imagen antes de ejecutar el borrado.")

    return background, mask


def run_renovation(
    editor_value,
    radius: int,
    method_name: str,
    dilate_iter: int,
    expand_px: int,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, object_mask = _extract_editor_payload(editor_value)
    bgr = utils.rgb_to_bgr(rgb)
    preview_mask = clean_mask(
        object_mask,
        dilate_iter=int(dilate_iter),
        expand_px=int(expand_px),
        feather_px=int(feather_px),
    )
    result, dilated_mask, backend_name = remove_and_fill(
        bgr,
        preview_mask,
        radius=int(radius),
        method=METHOD_MAP[method_name],
        dilate_iter=int(dilate_iter),
        expand_px=0,
        feather_px=0,
    )

    masked = bgr.copy()
    masked[dilated_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(bgr, 0.6, masked, 0.4, 0)
    result_rgb = utils.bgr_to_rgb(result)
    cv2.putText(
        result_rgb,
        f"Máscara manual -> {backend_name}",
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
