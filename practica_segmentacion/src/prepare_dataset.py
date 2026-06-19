from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from masks import generate_cell_mask, overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara imagenes y pseudo-mascaras.")
    parser.add_argument("--zip", default="dataset.zip", help="Ruta al zip del dataset.")
    parser.add_argument("--output", default="data", help="Directorio de salida.")
    parser.add_argument("--image-size", type=int, default=128, help="Tamano final cuadrado.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limite por clase. 0 usa todo.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_zip(zip_path: Path, raw_dir: Path) -> Path:
    dataset_dir = raw_dir / "bloodcells_dataset"
    if dataset_dir.exists():
        return dataset_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    return dataset_dir


def collect_images(dataset_dir: Path, max_per_class: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        images = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(images)
        if max_per_class > 0:
            images = images[:max_per_class]
        selected.extend(images)
    rng.shuffle(selected)
    return selected


def stratified_split(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_class[path.parent.name].append(path)

    splits = {"train": [], "val": [], "test": []}
    for class_paths in by_class.values():
        rng.shuffle(class_paths)
        n = len(class_paths)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits["train"].extend(class_paths[:n_train])
        splits["val"].extend(class_paths[n_train : n_train + n_val])
        splits["test"].extend(class_paths[n_train + n_val :])

    for split_paths in splits.values():
        rng.shuffle(split_paths)
    return splits


def write_processed_image(source: Path, image_path: Path, mask_path: Path, image_size: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer {source}")

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    mask = generate_cell_mask(image)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)


def save_examples(processed_dir: Path, outputs_dir: Path, limit: int = 12) -> None:
    examples_dir = outputs_dir / "mask_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((processed_dir / "images" / "train").glob("*.png"))[:limit]
    for image_path in images:
        mask_path = processed_dir / "masks" / "train" / image_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        overlay = overlay_mask(image, mask)
        panel = cv2.hconcat([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])
        cv2.imwrite(str(examples_dir / image_path.name), panel)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    zip_path = (base_dir / args.zip).resolve()
    output_dir = (base_dir / args.output).resolve()
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    outputs_dir = base_dir / "outputs"

    dataset_dir = extract_zip(zip_path, raw_dir)
    paths = collect_images(dataset_dir, args.max_per_class, args.seed)
    splits = stratified_split(paths, args.seed)

    summary = {
        "image_size": args.image_size,
        "max_per_class": args.max_per_class,
        "total_images": len(paths),
        "classes": dict(Counter(path.parent.name for path in paths)),
        "splits": {name: len(split_paths) for name, split_paths in splits.items()},
    }

    for split, split_paths in splits.items():
        for source in tqdm(split_paths, desc=f"Procesando {split}"):
            output_name = f"{source.parent.name}_{source.stem}.png"
            image_path = processed_dir / "images" / split / output_name
            mask_path = processed_dir / "masks" / split / output_name
            write_processed_image(source, image_path, mask_path, args.image_size)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_examples(processed_dir, outputs_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
