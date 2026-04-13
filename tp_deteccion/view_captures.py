import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visor de muestras guardadas por dataset_generator.py")
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "generated-files/captures_manifest.csv")
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Índice de la muestra en el manifest (por defecto -1, última).",
    )
    parser.add_argument("--sample-id", type=str, default="", help="Busca por sample_id exacto.")
    parser.add_argument("--label", type=int, default=-1, help="Filtra por label numérico.")
    parser.add_argument(
        "--list-last",
        type=int,
        default=10,
        help="Cantidad de filas recientes para listar en consola.",
    )
    return parser.parse_args()


def read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def select_row(rows: List[Dict[str, str]], sample_id: str, index: int, label: int) -> Optional[Dict[str, str]]:
    filtered = rows
    if label >= 0:
        filtered = [row for row in rows if int(row.get("label", "-999")) == label]
    if sample_id:
        for row in filtered:
            if row.get("sample_id", "") == sample_id:
                return row
        return None
    if not filtered:
        return None
    try:
        return filtered[index]
    except IndexError:
        return None


def try_load_image(path_text: str):
    image_path = Path(path_text)
    if not image_path.exists():
        return None
    return cv2.imread(str(image_path))


def main() -> None:
    args = parse_args()
    rows = read_manifest_rows(args.manifest)
    if not rows:
        raise FileNotFoundError(f"No hay manifest o está vacío: {args.manifest}")

    print(f"Manifest: {args.manifest}")
    print(f"Total muestras: {len(rows)}")
    print("Últimas filas:")
    for row in rows[-args.list_last :]:
        print(
            f"  sample_id={row.get('sample_id')} "
            f"label={row.get('label')} ({row.get('label_name')}) "
            f"area={row.get('area')}"
        )

    target = select_row(rows, args.sample_id, args.index, args.label)
    if target is None:
        raise ValueError("No se encontró muestra con esos filtros.")

    print("\nMuestra seleccionada:")
    print(f"  sample_id: {target.get('sample_id')}")
    print(f"  label: {target.get('label')} ({target.get('label_name')})")
    print(f"  area: {target.get('area')}")
    print(f"  hu: {[target.get(f'hu{i}') for i in range(1, 8)]}")
    print(f"  frame_path: {target.get('frame_path')}")
    print(f"  binary_path: {target.get('binary_path')}")
    print(f"  crop_path: {target.get('crop_path')}")

    frame = try_load_image(target.get("frame_path", ""))
    binary = try_load_image(target.get("binary_path", ""))
    crop = try_load_image(target.get("crop_path", ""))

    if frame is not None:
        cv2.imshow("capture - frame", frame)
    if binary is not None:
        cv2.imshow("capture - binary", binary)
    if crop is not None:
        cv2.imshow("capture - crop", crop)

    if frame is None and binary is None and crop is None:
        raise FileNotFoundError("No se pudieron abrir imágenes para la muestra seleccionada.")

    print("\nPresioná cualquier tecla en las ventanas para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
