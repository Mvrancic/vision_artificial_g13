from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from masks import overlay_mask
from model import bce_dice_loss, binary_iou, dice_coefficient


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera predicciones con la U-Net entrenada.")
    parser.add_argument("--model", default="outputs/unet_bloodcells.keras")
    parser.add_argument("--input", default="data/processed/images/test")
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def find_images(input_dir: Path, limit: int) -> list[Path]:
    image_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return image_paths[:limit]


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    model_path = (base_dir / args.model).resolve()
    input_dir = (base_dir / args.input).resolve()
    output_dir = (base_dir / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficient": dice_coefficient,
            "binary_iou": binary_iou,
        },
    )

    image_paths = find_images(input_dir, args.limit)
    if not image_paths:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No se encontraron imagenes en {input_dir}. "
            f"Extensiones soportadas: {extensions}"
        )

    saved = 0
    for path in image_paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"No se pudo leer {path}")
            continue
        resized = cv2.resize(image_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        prediction = model.predict(image_rgb[None, ...], verbose=0)[0, :, :, 0]
        mask = (prediction >= args.threshold).astype(np.uint8) * 255
        overlay = overlay_mask(resized, mask)
        panel = cv2.hconcat([resized, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])
        output_path = output_dir / f"{path.stem}_prediction.png"
        if cv2.imwrite(str(output_path), panel):
            saved += 1
        else:
            print(f"No se pudo guardar {output_path}")

    print(f"Predicciones guardadas en {output_dir}: {saved}/{len(image_paths)}")


if __name__ == "__main__":
    main()
