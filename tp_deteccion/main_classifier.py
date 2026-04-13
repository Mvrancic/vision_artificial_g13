import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np

from hu_shapes.dataset import load_labels_map
from hu_shapes.image_processor import ImageProcessor


COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(camera_index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TP Visión Artificial - Clasificador en tiempo real (Hu + DT).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", type=Path, default=Path("generated-files/modelo_formas.joblib"))
    p.add_argument("--labels", type=Path, default=Path("generated-files/labels.json"))
    p.add_argument("--debug", action="store_true", help="Muestra la imagen binaria/procesada.")
    return p.parse_args()


def load_model_bundle(model_path: Path, labels_path: Path) -> Tuple[object, Dict[int, str], Dict[int, np.ndarray], Dict[int, float]]:
    bundle = joblib.load(model_path)
    labels_map: Dict[int, str] = load_labels_map(labels_path)
    centroids: Dict[int, np.ndarray] = {}
    thresholds: Dict[int, float] = {}

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        stored_labels = bundle.get("labels_map", {})
        labels_map.update({int(k): str(v) for k, v in stored_labels.items()})
        centroids = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in bundle.get("class_centroids", {}).items()
        }
        thresholds = {
            int(k): float(v)
            for k, v in bundle.get("distance_thresholds", {}).items()
        }
        return model, labels_map, centroids, thresholds

    return bundle, labels_map, centroids, thresholds


def contour_distance_to_class(hu: np.ndarray, predicted_label: int, centroids: Dict[int, np.ndarray]) -> Optional[float]:
    centroid = centroids.get(predicted_label)
    if centroid is None:
        return None
    return float(np.linalg.norm(hu.reshape(-1) - centroid.reshape(-1)))


def main() -> None:
    args = parse_args()

    model, labels_map, centroids, thresholds = load_model_bundle(args.model, args.labels)

    processor = ImageProcessor(window_name="Main Classifier - Hu")
    processor.setup_ui()
    cv2.createTrackbar("Min prob (%)", processor.window_name, 60, 100, lambda v: None)
    cv2.createTrackbar("Dist tol (%)", processor.window_name, 120, 250, lambda v: None)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la webcam. En macOS: revisá permisos en System Settings > Privacy & Security > Camera "
            "para la app que ejecuta Python, reiniciá la app y reintentá."
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        params = processor.get_params()
        bin_img = processor.preprocess(frame, params)
        contours = processor.find_contours(bin_img, params)

        min_prob = int(cv2.getTrackbarPos("Min prob (%)", processor.window_name)) / 100.0
        dist_scale = int(cv2.getTrackbarPos("Dist tol (%)", processor.window_name)) / 100.0

        vis = frame.copy()
        for c in contours:
            hu = processor.contour_to_hu(c)
            sample = np.asarray(hu, dtype=np.float32).reshape(1, -1)

            pred = int(model.predict(sample)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample)[0]
                prob = float(np.max(proba))

            distance = contour_distance_to_class(hu, pred, centroids)
            max_distance = thresholds.get(pred)

            prob_ok = (prob is None) or (prob >= min_prob)
            distance_ok = (distance is None) or (max_distance is None) or (distance <= max_distance * dist_scale)
            ok_class = prob_ok and distance_ok
            color = COLOR_GREEN if ok_class else COLOR_RED

            name = labels_map.get(pred, str(pred))
            text = f"{name}" if prob is None else f"{name} ({prob:.2f})"
            if distance is not None:
                text = f"{text} d={distance:.2f}"
            if not ok_class:
                parts = []
                if prob is not None:
                    parts.append(f"p={prob:.2f}")
                if distance is not None:
                    parts.append(f"d={distance:.2f}")
                text = "NO-CLASIF" if not parts else f"NO-CLASIF ({', '.join(parts)})"

            processor.draw_contour_label(vis, c, text, color, 2)

        cv2.putText(
            vis,
            "[q] salir | verde=ok | rojo=prob/distancia baja",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(processor.window_name, vis)
        if args.debug:
            cv2.imshow("debug - binary", bin_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
