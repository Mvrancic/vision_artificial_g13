import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import sources
from common.trackbar import create_trackbar, get_trackbar_value
from enfoque3_reconstruccion.midas import DepthEstimator, colorize_depth

WINDOW = 'Enfoque 3 - Reconstructor de ambientes y planos 3D'


def detect_objects(depth_norm, near_pct, min_area_frac=0.01):
    """Detecta muebles como blobs cercanos (profundidad alta) y estima su volumen.

    near_pct: percentil [0..100]; pixeles mas cercanos que ese umbral son objetos.
    Devuelve lista de dicts con bbox, profundidad media y proxy de volumen.
    """
    h, w = depth_norm.shape
    thresh = np.percentile(depth_norm, near_pct)
    mask = (depth_norm >= thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h * w
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        region = depth_norm[y:y + bh, x:x + bw]
        mean_depth = float(region.mean())
        depth_span = float(region.max() - region.min())
        # proxy de volumen: area en pantalla * rango de profundidad del objeto
        volume = area * (depth_span + 0.05)
        objects.append({'bbox': (x, y, bw, bh), 'mean_depth': mean_depth,
                        'volume': volume, 'contour': cnt})
    objects.sort(key=lambda o: o['volume'], reverse=True)
    return objects, mask


def draw_objects(image, objects):
    out = image.copy()
    for i, obj in enumerate(objects):
        x, y, bw, bh = obj['bbox']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = '#%d d=%.2f vol=%.0f' % (i + 1, obj['mean_depth'], obj['volume'])
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def build_floorplan(image, depth_norm, objects, plan_size=400):
    """Vista de planta (top-down) back-proyectando los pixeles con la profundidad.

    Asume un modelo de camara pinhole simple (f ~ ancho). depth_norm 1=cerca.
    Convierte cada pixel a coordenadas mundo (X lateral, Z hacia adelante) y lo
    dibuja en una planta 2D, mas la huella de cada mueble detectado.
    """
    h, w = depth_norm.shape
    f = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    plan = np.full((plan_size, plan_size, 3), 30, dtype=np.uint8)

    # Z = distancia hacia adelante (lejos = Z grande). depth_norm cerca=1 -> Z chico.
    Z = (1.0 - depth_norm) * 4.0 + 0.5

    step = max(1, h // 160)  # submuestreo para que sea rapido
    for v in range(0, h, step):
        for u in range(0, w, step):
            z = Z[v, u]
            x = (u - cx) * z / f
            # mapear (x, z) a la planta: centro horizontal, lejos arriba
            px = int(plan_size / 2 + x * (plan_size / 8.0))
            py = int(plan_size - 1 - z * (plan_size / 5.0))
            if 0 <= px < plan_size and 0 <= py < plan_size:
                plan[py, px] = image[v, u]

    # huella de los muebles detectados, etiquetada
    for i, obj in enumerate(objects):
        bx, by, bw, bh = obj['bbox']
        u = bx + bw / 2.0
        v = by + bh
        z = Z[min(int(v), h - 1), min(int(u), w - 1)]
        x = (u - cx) * z / f
        px = int(plan_size / 2 + x * (plan_size / 8.0))
        py = int(plan_size - 1 - z * (plan_size / 5.0))
        if 0 <= px < plan_size and 0 <= py < plan_size:
            cv2.circle(plan, (px, py), 7, (0, 255, 0), 2)
            cv2.putText(plan, '#%d' % (i + 1), (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(plan, 'PLANTA (top-down)', (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return plan


def montage(original, depth_color, objects_img, plan, target_w=900):
    """Arma la grilla 2x2 con las cuatro vistas a un ancho fijo."""
    cell_w = target_w // 2
    def fit(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (cell_w, int(h * cell_w / w)))
    a, b, c = fit(original), fit(depth_color), fit(objects_img)
    # planta cuadrada al alto de las celdas
    d = cv2.resize(plan, (cell_w, a.shape[0]))
    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    # igualar anchos por si difieren 1px
    return np.vstack([top, bottom[:, :top.shape[1]]])


def run(image_path=None, use_camera=False):
    if use_camera:
        image = sources.grab_frame_from_camera()
        if image is None:
            print('Captura cancelada.')
            return
    else:
        if image_path is None:
            image_path = os.path.join(sources.REPO_IMAGES, 'road.jpg')
        image = sources.read_image(image_path)
    image = sources.resize_to_width(image, 640)

    print('Cargando estimador de profundidad...')
    estimator = DepthEstimator()
    print('Backend de profundidad:', estimator.backend)
    depth = estimator.predict(image)

    cv2.namedWindow(WINDOW)
    create_trackbar('Umbral cercania %', WINDOW, 99, initial=80)
    save_count = 0

    while True:
        near_pct = min(get_trackbar_value('Umbral cercania %', WINDOW), 99)
        objects, _ = detect_objects(depth, near_pct)
        depth_color = colorize_depth(depth)
        objects_img = draw_objects(image, objects)
        plan = build_floorplan(image, depth, objects)
        view = montage(image, depth_color, objects_img, plan)

        cv2.putText(view, 'backend: %s  | objetos: %d  (s=guardar q=salir)'
                    % (estimator.backend, len(objects)),
                    (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = os.path.join(os.path.dirname(__file__),
                                    'reconstruccion_%d.png' % save_count)
            cv2.imwrite(out_path, view)
            print('Guardado:', out_path)
            save_count += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
