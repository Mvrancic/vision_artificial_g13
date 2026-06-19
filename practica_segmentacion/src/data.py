from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def read_image(path: str | Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def read_mask(path: str | Path, image_size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"No se pudo leer la mascara: {path}")
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask[..., None]


def make_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    image_paths = sorted(Path(images_dir).glob("*.png"))
    mask_paths = [Path(masks_dir) / path.name for path in image_paths]

    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan mascaras para {len(missing)} imagenes.")

    def loader(image_path: tf.Tensor, mask_path: tf.Tensor):
        image, mask = tf.numpy_function(
            _load_pair,
            [image_path, mask_path, image_size],
            [tf.float32, tf.float32],
        )
        image.set_shape((image_size, image_size, 3))
        mask.set_shape((image_size, image_size, 1))
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in image_paths], [str(path) for path in mask_paths])
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    return dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _load_pair(image_path: bytes, mask_path: bytes, image_size: np.int64):
    size = int(image_size)
    image = read_image(image_path.decode("utf-8"), size)
    mask = read_mask(mask_path.decode("utf-8"), size)
    return image, mask
