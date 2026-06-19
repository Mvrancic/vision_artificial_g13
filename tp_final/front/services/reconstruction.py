from __future__ import annotations

import numpy as np

from tp_final.enfoque3_reconstruccion.main import (
    PROC_WIDTH,
    build_floorplan,
    detect_objects,
    draw_objects,
    montage,
)
from tp_final.enfoque3_reconstruccion.midas import DepthEstimator, colorize_depth
from tp_final.front import utils


_ESTIMATOR: DepthEstimator | None = None


def get_estimator() -> DepthEstimator:
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = DepthEstimator()
    return _ESTIMATOR


def process_reconstruction(
    image: np.ndarray,
    near_pct: int,
    plan_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        raise ValueError("No hay imagen cargada.")

    frame = utils.resize_for_display(rgb, PROC_WIDTH)
    bgr = utils.rgb_to_bgr(frame)
    estimator = get_estimator()
    depth = estimator.predict(bgr)
    objects = detect_objects(depth, int(near_pct))
    depth_vis = colorize_depth(depth)
    objects_img = draw_objects(bgr, objects)
    plan = build_floorplan(bgr, depth, objects, mode=plan_mode)
    summary = montage(bgr, depth_vis, objects_img, plan)

    summary_text = (
        f"Backend: {estimator.backend} | "
        f"Objetos detectados: {len(objects)} | "
        f"Modo de planta: {plan_mode}"
    )

    return (
        utils.bgr_to_rgb(summary),
        utils.bgr_to_rgb(bgr),
        utils.bgr_to_rgb(depth_vis),
        utils.bgr_to_rgb(objects_img),
        utils.bgr_to_rgb(plan),
        summary_text,
    )
