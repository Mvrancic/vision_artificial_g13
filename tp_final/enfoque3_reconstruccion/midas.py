import os
import urllib.request
import cv2
import numpy as np


# Estimacion de profundidad monocular con MiDaS (modelo small) corriendo sobre
# cv2.dnn -> NO requiere torch ni tensorflow, solo opencv. El modelo ONNX se
# descarga bajo demanda. Si no hay red / no se puede cargar, se usa una
# profundidad heuristica (mas abajo en la imagen = mas cerca) para que el TP
# siga funcionando.

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model-small.onnx')
MODEL_URL = 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx'

# Normalizacion de MiDaS (ImageNet)
INPUT_SIZE = 256
MEAN = (0.485, 0.456, 0.406)


def download_model():
    """Descarga el ONNX de MiDaS small si no esta en disco. Devuelve True/False."""
    if os.path.exists(MODEL_PATH):
        return True
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        print('Descargando modelo MiDaS small (~67MB)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('Modelo guardado en', MODEL_PATH)
        return True
    except Exception as exc:  # sin red, etc.
        print('No se pudo descargar MiDaS:', exc)
        return False


class DepthEstimator:
    def __init__(self):
        self.net = None
        self.backend = 'heuristica'
        if download_model():
            try:
                self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
                self.backend = 'MiDaS (cv2.dnn)'
            except Exception as exc:
                print('No se pudo cargar el modelo ONNX:', exc)
                self.net = None

    def _predict_midas(self, image):
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
            (MEAN[0] * 255, MEAN[1] * 255, MEAN[2] * 255),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, H, W)
        depth = out[0]
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        return depth

    def _predict_heuristic(self, image):
        # gradiente vertical: la base de la imagen (piso cercano) = mas cerca
        h, w = image.shape[:2]
        rows = np.linspace(0.0, 1.0, h, dtype=np.float32)
        return np.tile(rows[:, None], (1, w))

    def predict(self, image):
        """Devuelve un mapa de profundidad relativo normalizado a [0,1] (1 = cerca)."""
        if self.net is not None:
            raw = self._predict_midas(image)
        else:
            raw = self._predict_heuristic(image)
        d_min, d_max = float(raw.min()), float(raw.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(raw)
        norm = (raw - d_min) / (d_max - d_min)
        return norm.astype(np.float32)


def colorize_depth(depth_norm):
    """Mapa de profundidad a color para visualizar (cerca=calido)."""
    vis = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
