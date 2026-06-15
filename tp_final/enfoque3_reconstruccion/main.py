import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import sources
from common.trackbar import create_trackbar, get_trackbar_value
from enfoque3_reconstruccion.midas import DepthEstimator, colorize_depth

WINDOW = 'Enfoque 3 - Reconstructor de ambientes y planos 3D'

PROC_WIDTH = 480  # ancho al que procesamos cada cuadro (para que el video corra fluido)


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
        volume = area * (depth_span + 0.05)
        objects.append({'bbox': (x, y, bw, bh), 'mean_depth': mean_depth,
                        'volume': volume})
    objects.sort(key=lambda o: o['volume'], reverse=True)
    return objects


def depth_label(mean_depth):
    """Traduce la profundidad relativa a una etiqueta entendible."""
    if mean_depth >= 0.66:
        return 'CERCA'
    if mean_depth >= 0.4:
        return 'MEDIO'
    return 'LEJOS'


def draw_objects(image, objects):
    out = image.copy()
    for i, obj in enumerate(objects):
        x, y, bw, bh = obj['bbox']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = '#%d %s (d=%.2f)' % (i + 1, depth_label(obj['mean_depth']), obj['mean_depth'])
        cv2.putText(out, label, (x + 3, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, label, (x + 3, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def build_floorplan(image, depth_norm, objects, plan_size=420, mode='heat'):
    """Vista de planta (top-down): back-proyecta los pixeles con la profundidad.

    Modelo de camara pinhole simple (f ~ ancho). La camara esta abajo al centro y
    mira hacia "arriba" del plano (lejos). Eje X = izquierda/derecha, eje vertical
    del plano = distancia hacia adelante. La escala es RELATIVA (MiDaS da profundidad
    relativa, no metros). Se dibuja grilla, ejes, la camara con su cono de vision y
    la huella de cada mueble detectado.

    mode='heat': grilla de ocupacion estilo radar (acumula densidad -> superficies
    solidas y nitidas). mode='color': nube de puntos con el color real (mas ruidosa).
    """
    h, w = depth_norm.shape
    f = 0.9 * w
    cx = w / 2.0
    cam_x = plan_size / 2.0           # camara: centro horizontal, base del plano
    cam_y = plan_size - 12
    sx = plan_size / 8.0              # escala horizontal (X mundo -> pixeles plano)
    sy = plan_size / 5.0              # escala en profundidad

    plan = np.full((plan_size, plan_size, 3), 28, dtype=np.uint8)

    # --- grilla de fondo ---
    grid_color = (55, 55, 55)
    for g in range(0, plan_size, plan_size // 8):
        cv2.line(plan, (g, 0), (g, plan_size), grid_color, 1)
        cv2.line(plan, (0, g), (plan_size, g), grid_color, 1)

    # --- back-proyeccion (vectorizado para que corra en video) ---
    Z = (1.0 - depth_norm) * 4.0 + 0.5   # lejos = Z grande
    step = max(1, h // 180)
    vs = np.arange(0, h, step)
    us = np.arange(0, w, step)
    uu, vv = np.meshgrid(us, vs)
    zz = Z[vv, uu]
    xx = (uu - cx) * zz / f
    px = (cam_x + xx * sx).astype(np.int32).ravel()
    py = (cam_y - zz * sy).astype(np.int32).ravel()
    ok = (px >= 0) & (px < plan_size) & (py >= 0) & (py < plan_size)
    px, py = px[ok], py[ok]

    if mode == 'color':
        # nube de puntos con el color real de la imagen
        cols = image[vv, uu].reshape(-1, 3)[ok]
        plan[py, px] = cols
    else:
        # grilla de ocupacion: cuento cuantos puntos caen en cada celda. Las
        # superficies verticales (paredes, muebles) acumulan muchos -> brillan;
        # el ruido y el piso quedan tenues. Escala log + colormap tipo radar.
        acc = np.zeros((plan_size, plan_size), np.float32)
        np.add.at(acc, (py, px), 1.0)
        acc = cv2.GaussianBlur(acc, (0, 0), 1.2)   # splat: rellena huecos
        peak = float(acc.max())
        if peak > 0:
            norm = np.log1p(acc) / np.log1p(peak)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            m = acc > 0.2
            plan[m] = heat[m]

    # --- camara + cono de vision ---
    cv2.circle(plan, (int(cam_x), int(cam_y)), 5, (0, 255, 255), -1)
    half = (w / 2.0) / f
    for sign in (-1, 1):
        far_z = 4.5
        fx = cam_x + sign * half * far_z * sx
        fy = cam_y - far_z * sy
        cv2.line(plan, (int(cam_x), int(cam_y)), (int(fx), int(fy)), (0, 120, 120), 1)

    # --- huella de los muebles detectados ---
    for i, obj in enumerate(objects):
        bx, by, bw, bh = obj['bbox']
        u = bx + bw / 2.0
        v = by + bh
        z = Z[min(int(v), h - 1), min(int(u), w - 1)]
        x = (u - cx) * z / f
        ox = int(cam_x + x * sx)
        oy = int(cam_y - z * sy)
        if 0 <= ox < plan_size and 0 <= oy < plan_size:
            cv2.circle(plan, (ox, oy), 8, (0, 255, 0), 2)
            cv2.putText(plan, '#%d' % (i + 1), (ox + 10, oy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- titulos / ejes / escala ---
    def txt(s, org, color=(255, 255, 255), scale=0.45):
        cv2.putText(plan, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(plan, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    txt('PLANTA (vista desde arriba)', (8, 20), (255, 255, 255), 0.5)
    txt('LEJOS', (plan_size // 2 - 28, 40), (180, 180, 255))
    txt('CAMARA', (int(cam_x) - 30, plan_size - 18), (0, 255, 255))
    txt('izq', (6, plan_size // 2))
    txt('der', (plan_size - 40, plan_size // 2))
    txt('escala relativa', (8, plan_size - 8), (160, 160, 160), 0.4)
    return plan


def montage(original, depth_color, objects_img, plan, target_w=920):
    """Arma la grilla 2x2 con las cuatro vistas a un ancho fijo."""
    cell_w = target_w // 2

    def fit(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (cell_w, int(h * cell_w / w)))

    a, b, c = fit(original), fit(depth_color), fit(objects_img)
    d = cv2.resize(plan, (cell_w, a.shape[0]))
    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    return np.vstack([top, bottom[:, :top.shape[1]]])


def run(image_path=None, use_camera=False):
    print('Cargando estimador de profundidad...')
    estimator = DepthEstimator()
    print('Backend de profundidad:', estimator.backend)

    cap = None
    static_frame = None
    static_depth = None
    if use_camera:
        cap = sources.open_camera()
    else:
        if image_path is None:
            image_path = os.path.join(sources.REPO_IMAGES, 'road.jpg')
        static_frame = sources.resize_to_width(sources.read_image(image_path), PROC_WIDTH)
        static_depth = estimator.predict(static_frame)  # se calcula una sola vez

    cv2.namedWindow(WINDOW)
    create_trackbar('Umbral cercania %', WINDOW, 99, initial=80)
    save_count = 0
    plan_mode = 'heat'   # 'heat' = radar/ocupacion (nitido), 'color' = nube real
    tick_freq = cv2.getTickFrequency()
    last_tick = cv2.getTickCount()
    fps = 0.0

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            frame = sources.resize_to_width(frame, PROC_WIDTH)
            depth = estimator.predict(frame)      # MiDaS en vivo, cuadro a cuadro
        else:
            frame = static_frame
            depth = static_depth

        near_pct = min(get_trackbar_value('Umbral cercania %', WINDOW), 99)
        objects = detect_objects(depth, near_pct)
        view = montage(frame, colorize_depth(depth), draw_objects(frame, objects),
                       build_floorplan(frame, depth, objects, mode=plan_mode))

        # FPS (cv2.getTickCount, no usa reloj de python)
        now = cv2.getTickCount()
        dt = (now - last_tick) / tick_freq
        last_tick = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        info = 'backend: %s | objetos: %d | %.1f fps | planta: %s  (c=planta s=guardar q=salir)' % (
            estimator.backend, len(objects), fps, plan_mode)
        cv2.putText(view, info, (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, info, (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            plan_mode = 'color' if plan_mode == 'heat' else 'heat'
        elif key == ord('s'):
            out_path = os.path.join(os.path.dirname(__file__),
                                    'reconstruccion_%d.png' % save_count)
            cv2.imwrite(out_path, view)
            print('Guardado:', out_path)
            save_count += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(use_camera=True)
