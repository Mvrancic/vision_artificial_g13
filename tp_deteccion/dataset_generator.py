import argparse
from datetime import datetime
from pathlib import Path

import cv2

from hu_shapes.dataset import DatasetRow, append_row_csv, load_labels_map, save_capture_record, save_labels_map
from hu_shapes.image_processor import ImageProcessor

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TP Visión Artificial - Generador de descriptores (Hu Moments).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=Path, default=BASE_DIR / "generated-files/dataset_hu.csv")
    p.add_argument("--labels", type=Path, default=BASE_DIR / "generated-files/labels.json")
    p.add_argument("--captures-dir", type=Path, default=BASE_DIR / "generated-files/captures")
    p.add_argument("--captures-manifest", type=Path, default=BASE_DIR / "generated-files/captures_manifest.csv")
    p.add_argument(
        "--no-save-captures",
        action="store_true",
        help="No guarda fotos de cada muestra, solo los descriptores en CSV.",
    )
    p.add_argument("--debug", action="store_true", help="Muestra la imagen binaria/procesada.")
    return p.parse_args()

def open_camera(camera_index: int) -> cv2.VideoCapture:
    """
    macOS: AVFoundation suele ser el backend más estable.
    Si falla, probamos apertura genérica.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(camera_index)
    return cap


def main() -> None:
    args = parse_args()

    processor = ImageProcessor(window_name="Dataset Generator - Hu")
    processor.setup_ui()

    label_window = "Dataset Generator - Hu"
    cv2.createTrackbar("Label ID", label_window, 1, 99, lambda v: None)

    labels_map = load_labels_map(args.labels)
    if not labels_map:
        # default placeholders (editable por el usuario)
        labels_map = {1: "clase_1", 2: "clase_2", 3: "clase_3"}
        save_labels_map(args.labels, labels_map)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la webcam. En macOS: asegurate de dar permiso de Cámara "
            "a la app que ejecuta Python (Terminal/iTerm/Cursor) en System Settings > Privacy & Security > Camera, "
            "cerrá y reabrí la app y volvé a ejecutar."
        )

    last_saved = 0
    last_message = "Buscá un solo objeto bien contrastado y presioná SPACE para guardar."
    run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        params = processor.get_params()
        bin_img = processor.preprocess(frame, params)
        contours = processor.find_contours(bin_img, params)
        primary = processor.pick_primary_contour(contours)

        label_id = int(cv2.getTrackbarPos("Label ID", label_window))
        label_name = labels_map.get(label_id, f"id_{label_id}")

        vis = frame.copy()
        if primary is not None:
            area = int(cv2.contourArea(primary))
            processor.draw_contour_label(vis, primary, f"DET: {label_id} ({label_name}) area={area}", (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"[SPACE] guardar | [v] ver ultima | Label={label_id} ({label_name}) | guardados={last_saved} | [q] salir",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"contornos={len(contours)} | {last_message}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(label_window, vis)
        if args.debug:
            cv2.imshow("debug - binary", bin_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            if args.no_save_captures:
                last_message = "Guardado de fotos desactivado (--no-save-captures)."
            else:
                last_capture = Path(args.captures_dir) / "last_preview_crop.png"
                if last_capture.exists():
                    preview = cv2.imread(str(last_capture))
                    if preview is not None:
                        cv2.imshow("ultima muestra - crop", preview)
                else:
                    last_message = "Todavía no hay fotos guardadas."
        if key == ord(" "):
            if primary is None:
                last_message = "No hay contorno valido. Ajustá threshold, blur, morfología o area mínima."
                continue
            hu = processor.contour_to_hu(primary)
            append_row_csv(args.out, DatasetRow(label=label_id, hu=hu))
            last_saved += 1
            sample_id = f"{run_prefix}_{last_saved:04d}"

            if not args.no_save_captures:
                record = save_capture_record(
                    captures_dir=args.captures_dir,
                    manifest_path=args.captures_manifest,
                    sample_id=sample_id,
                    label=label_id,
                    label_name=label_name,
                    frame_bgr=frame,
                    binary_img=bin_img,
                    contour=primary,
                    hu=hu,
                )
                # último preview rápido para abrir con la tecla "v"
                preview_path = Path(args.captures_dir) / "last_preview_crop.png"
                preview = cv2.imread(str(record.crop_path))
                if preview is not None:
                    cv2.imwrite(str(preview_path), preview)

                last_message = (
                    f"Muestra {sample_id} guardada ({label_name}). "
                    f"Fotos en {args.captures_dir} y manifest en {args.captures_manifest}."
                )
            else:
                last_message = f"Muestra {sample_id} guardada para label {label_id} ({label_name})."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
