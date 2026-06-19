from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    from simple_lama_inpainting.utils.util import download_model, prepare_img_and_mask
except Exception:  # pragma: no cover
    LAMA_MODEL_URL = None
    download_model = None
    prepare_img_and_mask = None

try:
    from rembg import new_session, remove
except Exception:  # pragma: no cover
    new_session = None
    remove = None


@dataclass
class InpaintResult:
    image: np.ndarray
    mask: np.ndarray
    backend_name: str


_LAMA_MODEL = None
_LAMA_ERROR: str | None = None
_REMBG_SESSIONS: dict[str, object] = {}
_REMBG_ERROR: str | None = None
TP_FINAL_DIR = Path(__file__).resolve().parents[1]
LAMA_CACHE_DIR = TP_FINAL_DIR / ".cache" / "torch"
CACHE_DIR = TP_FINAL_DIR / ".cache"
U2NET_CACHE_DIR = CACHE_DIR / "u2net"


def configure_model_cache() -> None:
    os.environ.setdefault("TORCH_HOME", str(LAMA_CACHE_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("U2NET_HOME", str(U2NET_CACHE_DIR))
    LAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    U2NET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _component_closest_to_center(mask: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    cx, cy = center_xy
    best_label = 0
    best_score = float("-inf")
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        px, py = centroids[label]
        distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        score = area - (distance * 8.0)
        if score > best_score:
            best_label = label
            best_score = score

    if best_label == 0:
        return _largest_component(mask)
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _component_best_overlap(mask: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    roi = np.zeros_like(mask, dtype=np.uint8)
    roi[y : y + h, x : x + w] = 1
    best_label = 0
    best_overlap = 0
    best_area = 0
    for label in range(1, num_labels):
        component = labels == label
        overlap = int((component & roi.astype(bool)).sum())
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap > best_overlap or (overlap == best_overlap and area > best_area):
            best_label = label
            best_overlap = overlap
            best_area = area

    if best_label == 0:
        return _largest_component(mask)
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _mask_score(mask: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return -1.0

    mask_bin = mask > 0
    area = float(mask_bin.sum())
    if area <= 0:
        return -1.0

    rect_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    rect_mask[y : y + h, x : x + w] = 1
    intersection = float((mask_bin & rect_mask.astype(bool)).sum())
    union = float((mask_bin | rect_mask.astype(bool)).sum())
    iou = intersection / max(union, 1.0)

    crop = mask_bin[y : y + h, x : x + w]
    coverage = float(crop.sum()) / max(float(w * h), 1.0)
    full_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    area_ratio = area / max(full_area, 1.0)
    edge_touch = float(
        mask_bin[0, :].any() or mask_bin[-1, :].any() or mask_bin[:, 0].any() or mask_bin[:, -1].any()
    )

    penalty = 0.0
    if coverage < 0.08:
        penalty += 0.3
    if area_ratio > 0.65:
        penalty += 0.5
    return (iou * 0.45) + (coverage * 0.45) - (edge_touch * 0.1) - penalty


def _expand_rect(rect: tuple[int, int, int, int], shape: tuple[int, int], pad_px: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    height, width = shape[:2]
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(width, x + w + pad_px)
    y1 = min(height, y + h + pad_px)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _expand_rect_fraction(
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad_left: float,
    pad_top: float,
    pad_right: float,
    pad_bottom: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    height, width = shape[:2]
    x0 = max(0, int(round(x - (w * pad_left))))
    y0 = max(0, int(round(y - (h * pad_top))))
    x1 = min(width, int(round(x + w + (w * pad_right))))
    y1 = min(height, int(round(y + h + (h * pad_bottom))))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _paste_crop_mask(
    crop_mask: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    x, y, w, h = crop_rect
    canvas = np.zeros(full_shape, dtype=np.uint8)
    canvas[y : y + h, x : x + w] = crop_mask[:h, :w]
    return canvas


def _run_grabcut_crop(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    crop_rect = _expand_rect_fraction(rect, image.shape[:2], 0.18, 0.18, 0.18, 0.22)
    cx, cy, cw, ch = crop_rect
    crop = image[cy : cy + ch, cx : cx + cw]
    local_rect = (
        max(1, rect[0] - cx),
        max(1, rect[1] - cy),
        min(rect[2], max(2, cw - 2)),
        min(rect[3], max(2, ch - 2)),
    )

    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, local_rect, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_RECT)
    object_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    center = (local_rect[0] + (local_rect[2] / 2.0), local_rect[1] + (local_rect[3] / 2.0))
    object_mask = _component_closest_to_center(object_mask, center)
    return _paste_crop_mask(object_mask, crop_rect, image.shape[:2])


def get_rembg_session(model_name: str):
    global _REMBG_ERROR
    if model_name in _REMBG_SESSIONS:
        return _REMBG_SESSIONS[model_name]
    if new_session is None or remove is None:
        _REMBG_ERROR = "rembg no está instalado en el entorno actual."
        raise RuntimeError(_REMBG_ERROR)

    configure_model_cache()
    try:
        session = new_session(model_name=model_name, providers=["CPUExecutionProvider"])
        _REMBG_SESSIONS[model_name] = session
        return session
    except Exception as exc:  # pragma: no cover
        _REMBG_ERROR = f"No se pudo inicializar rembg ({model_name}): {exc}"
        raise RuntimeError(_REMBG_ERROR) from exc


def _run_rembg_crop(image: np.ndarray, rect: tuple[int, int, int, int], model_name: str) -> np.ndarray:
    if model_name == "u2net_human_seg":
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        session = get_rembg_session(model_name)
        mask = remove(rgb_image, session=session, only_mask=True, post_process_mask=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = np.asarray(mask, dtype=np.uint8)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return _component_best_overlap(mask, rect)

    crop_rect = _expand_rect_fraction(rect, image.shape[:2], 0.30, 0.25, 0.30, 0.35)
    cx, cy, cw, ch = crop_rect
    crop = image[cy : cy + ch, cx : cx + cw]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    session = get_rembg_session(model_name)
    mask = remove(rgb_crop, session=session, only_mask=True, post_process_mask=True)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.asarray(mask, dtype=np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    local_rect = (rect[0] - cx, rect[1] - cy, rect[2], rect[3])
    center = (local_rect[0] + (local_rect[2] / 2.0), local_rect[1] + (local_rect[3] / 2.0))
    mask = _component_closest_to_center(mask, center)
    return _paste_crop_mask(mask, crop_rect, image.shape[:2])


def segment_object(image: np.ndarray, rect: tuple[int, int, int, int], mode: str = "auto") -> tuple[np.ndarray, str]:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise ValueError("La ROI no es válida.")

    candidates: list[tuple[str, np.ndarray]] = []

    try:
        if mode in {"auto", "grabcut"}:
            candidates.append(("GrabCut", _run_grabcut_crop(image, rect)))
    except Exception:
        pass

    try:
        if mode in {"auto", "rembg_general"}:
            candidates.append(("Rembg-General", _run_rembg_crop(image, rect, "u2net")))
    except Exception:
        pass

    try:
        if mode in {"auto", "rembg_human"}:
            candidates.append(("Rembg-Human", _run_rembg_crop(image, rect, "u2net_human_seg")))
    except Exception:
        pass

    if not candidates:
        raise RuntimeError("No se pudo segmentar el objeto con ninguno de los backends disponibles.")

    best_name, best_mask = max(candidates, key=lambda item: _mask_score(item[1], rect))
    if mode == "auto" and len(candidates) >= 2:
        name_to_mask = {name: mask for name, mask in candidates}
        if "GrabCut" in name_to_mask and best_name != "GrabCut":
            merged = cv2.bitwise_or(best_mask, name_to_mask["GrabCut"])
            merged_score = _mask_score(merged, rect)
            if merged_score >= (_mask_score(best_mask, rect) - 0.03):
                best_name = f"{best_name} + GrabCut"
                best_mask = merged

    return best_mask, best_name


def clean_mask(
    object_mask: np.ndarray,
    dilate_iter: int,
    smooth_radius: int = 5,
    expand_px: int = 0,
    feather_px: int = 0,
) -> np.ndarray:
    mask = (object_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=max(int(dilate_iter), 0))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled

    blur_size = max(int(smooth_radius), 1)
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask = (mask > 32).astype(np.uint8) * 255

    if expand_px > 0:
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, grow, iterations=int(expand_px))

    if feather_px > 0:
        feather_size = max(1, int(feather_px) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        mask = (mask > 12).astype(np.uint8) * 255

    return _largest_component(mask)


def inpaint_opencv(image: np.ndarray, mask: np.ndarray, radius: int, method: int) -> InpaintResult:
    flag = cv2.INPAINT_TELEA if method == 0 else cv2.INPAINT_NS
    result = cv2.inpaint(image, mask, max(int(radius), 1), flag)
    backend_name = "OpenCV-TELEA" if method == 0 else "OpenCV-Navier-Stokes"
    return InpaintResult(image=result, mask=mask, backend_name=backend_name)


def get_lama_model():
    global _LAMA_MODEL
    global _LAMA_ERROR
    if _LAMA_MODEL is not None:
        return _LAMA_MODEL
    if _LAMA_ERROR is not None:
        raise RuntimeError(_LAMA_ERROR)
    if download_model is None or prepare_img_and_mask is None:
        _LAMA_ERROR = "simple-lama-inpainting no está instalado."
        raise RuntimeError(_LAMA_ERROR)
    try:
        configure_model_cache()
        model_path = download_model(LAMA_MODEL_URL)
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        _LAMA_MODEL = model
        return _LAMA_MODEL
    except Exception as exc:  # pragma: no cover
        _LAMA_ERROR = f"No se pudo inicializar LaMa: {exc}"
        raise RuntimeError(_LAMA_ERROR) from exc


def inpaint_lama(image: np.ndarray, mask: np.ndarray) -> InpaintResult:
    lama = get_lama_model()
    original_h, original_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    pil_mask = Image.fromarray(mask)
    image_tensor, mask_tensor = prepare_img_and_mask(pil_image, pil_mask, torch.device("cpu"))
    with torch.inference_mode():
        inpainted = lama(image_tensor, mask_tensor)
        cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
        cur_res = cur_res[:original_h, :original_w]
    result_bgr = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    return InpaintResult(image=result_bgr, mask=mask, backend_name="LaMa")


def backend_available(method_name: str) -> tuple[bool, str | None]:
    if method_name != "LaMa":
        return True, None
    if download_model is None or prepare_img_and_mask is None:
        return False, "LaMa no está instalado en el entorno actual."
    if _LAMA_ERROR is not None:
        return False, _LAMA_ERROR
    return True, None
