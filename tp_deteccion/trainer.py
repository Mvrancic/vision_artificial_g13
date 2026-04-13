import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from hu_shapes.dataset import load_dataset_csv, load_labels_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TP Visión Artificial - Entrenador (DecisionTree + Hu).")
    p.add_argument("--dataset", type=Path, default=Path("generated-files/dataset_hu.csv"))
    p.add_argument("--labels", type=Path, default=Path("generated-files/labels.json"))
    p.add_argument("--out-model", type=Path, default=Path("generated-files/modelo_formas.joblib"))
    p.add_argument("--out-tree", type=Path, default=Path("generated-files/decision_tree.png"))
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    return p.parse_args()


def build_rejection_stats(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    centroids: Dict[int, List[float]] = {}
    thresholds: Dict[int, float] = {}

    for cls in np.unique(y):
        cls_int = int(cls)
        samples = X[y == cls]
        centroid = samples.mean(axis=0)
        distances = np.linalg.norm(samples - centroid, axis=1)

        base_threshold = float(np.max(distances)) if len(distances) else 0.0
        robust_threshold = float(np.mean(distances) + 2.0 * np.std(distances)) if len(distances) else 0.0
        thresholds[cls_int] = max(base_threshold, robust_threshold, 0.35)
        centroids[cls_int] = centroid.astype(np.float32).tolist()

    return centroids, thresholds


def maybe_print_validation(X: np.ndarray, y: np.ndarray, max_depth: int, min_samples_leaf: int, labels_map: Dict[int, str]) -> None:
    class_counts = Counter(y.tolist())
    if len(class_counts) < 2 or min(class_counts.values()) < 2 or len(y) < 8:
        print("Validación omitida: hacen falta al menos 2 muestras por clase y un dataset un poco más grande.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    probe_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    probe_model.fit(X_train, y_train)
    y_pred = probe_model.predict(X_test)

    print(f"Accuracy hold-out: {accuracy_score(y_test, y_pred):.3f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[labels_map.get(int(cls), str(int(cls))) for cls in probe_model.classes_],
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()

    X, y = load_dataset_csv(args.dataset)
    if X.shape[0] < 3:
        raise ValueError("Dataset muy chico. Capturá más muestras antes de entrenar.")
    unique_classes = np.unique(y)
    if unique_classes.shape[0] < 2:
        raise ValueError(
            "El dataset tiene una sola clase. Para entrenar el clasificador necesitás al menos 2 etiquetas distintas."
        )

    labels_map = load_labels_map(args.labels)
    class_counts = Counter(y.tolist())
    print("Distribución de clases:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls} ({labels_map.get(int(cls), str(int(cls)))}): {count} muestras")

    maybe_print_validation(X, y, args.max_depth, args.min_samples_leaf, labels_map)

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    clf.fit(X, y)
    centroids, thresholds = build_rejection_stats(X, y)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels_map": {int(k): str(v) for k, v in labels_map.items()},
            "class_centroids": centroids,
            "distance_thresholds": thresholds,
            "feature_names": [f"hu{i}" for i in range(1, 8)],
        },
        args.out_model,
    )

    class_names = []
    for c in clf.classes_:
        class_names.append(labels_map.get(int(c), str(int(c))))

    plt.figure(figsize=(18, 9))
    plot_tree(
        clf,
        feature_names=[f"hu{i}" for i in range(1, 8)],
        class_names=class_names if class_names else None,
        filled=True,
        rounded=True,
        impurity=True,
    )
    args.out_tree.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_tree, dpi=200)
    plt.close()

    print(f"Modelo guardado en: {args.out_model}")
    print(f"Árbol exportado en: {args.out_tree}")


if __name__ == "__main__":
    main()
