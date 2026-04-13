import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CSV_HEADER: List[str] = ["label"] + [f"hu{i}" for i in range(1, 8)]
CAPTURES_HEADER: List[str] = [
    "sample_id",
    "label",
    "label_name",
    "area",
    "frame_path",
    "binary_path",
    "crop_path",
] + [f"hu{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class DatasetRow:
    label: int
    hu: np.ndarray


@dataclass(frozen=True)
class CaptureRecord:
    sample_id: str
    label: int
    label_name: str
    area: float
    frame_path: Path
    binary_path: Path
    crop_path: Path
    hu: np.ndarray


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_row_csv(csv_path: Path, row: DatasetRow) -> None:
    ensure_parent_dir(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as file_obj:
        writer = csv.writer(file_obj)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow([int(row.label)] + [float(value) for value in row.hu.reshape(-1).tolist()])


def load_dataset_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty dataset: {csv_path}")
        if header != CSV_HEADER:
            raise ValueError(f"Unexpected CSV header in {csv_path}. Expected {CSV_HEADER} but got {header}.")

        x_rows: List[List[float]] = []
        y_rows: List[int] = []
        for row in reader:
            if not row:
                continue
            y_rows.append(int(row[0]))
            x_rows.append([float(value) for value in row[1:8]])

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32)


def load_labels_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: Dict[int, str] = {}
    for key, value in raw.items():
        output[int(key)] = str(value)
    return output


def save_labels_map(path: Path, labels: Dict[int, str]) -> None:
    ensure_parent_dir(path)
    normalized = {str(int(key)): str(value) for key, value in sorted(labels.items(), key=lambda item: int(item[0]))}
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return normalized.strip("_") or "sin_nombre"


def extract_contour_crop(frame_bgr: np.ndarray, contour: np.ndarray, padding: int = 12) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(frame_bgr.shape[1], x + w + padding)
    y1 = min(frame_bgr.shape[0], y + h + padding)
    return frame_bgr[y0:y1, x0:x1].copy()


def save_capture_record(
    captures_dir: Path,
    manifest_path: Path,
    sample_id: str,
    label: int,
    label_name: str,
    frame_bgr: np.ndarray,
    binary_img: np.ndarray,
    contour: np.ndarray,
    hu: np.ndarray,
) -> CaptureRecord:
    label_dir = captures_dir / safe_name(label_name)
    label_dir.mkdir(parents=True, exist_ok=True)

    frame_path = label_dir / f"{sample_id}_frame.png"
    binary_path = label_dir / f"{sample_id}_binary.png"
    crop_path = label_dir / f"{sample_id}_crop.png"

    crop = extract_contour_crop(frame_bgr, contour)
    cv2.imwrite(str(frame_path), frame_bgr)
    cv2.imwrite(str(binary_path), binary_img)
    cv2.imwrite(str(crop_path), crop)

    record = CaptureRecord(
        sample_id=sample_id,
        label=label,
        label_name=label_name,
        area=float(cv2.contourArea(contour)),
        frame_path=frame_path,
        binary_path=binary_path,
        crop_path=crop_path,
        hu=np.asarray(hu, dtype=np.float32).reshape(-1),
    )
    append_capture_manifest(manifest_path, record)
    return record


def append_capture_manifest(manifest_path: Path, record: CaptureRecord) -> None:
    ensure_parent_dir(manifest_path)
    file_exists = manifest_path.exists()

    with manifest_path.open("a", newline="") as file_obj:
        writer = csv.writer(file_obj)
        if not file_exists:
            writer.writerow(CAPTURES_HEADER)
        writer.writerow(
            [
                record.sample_id,
                int(record.label),
                record.label_name,
                float(record.area),
                str(record.frame_path),
                str(record.binary_path),
                str(record.crop_path),
            ]
            + [float(value) for value in record.hu.tolist()]
        )


def load_capture_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def find_capture_by_sample_id(manifest_path: Path, sample_id: str) -> Optional[Dict[str, str]]:
    for row in load_capture_manifest(manifest_path):
        if row.get("sample_id") == sample_id:
            return row
    return None
