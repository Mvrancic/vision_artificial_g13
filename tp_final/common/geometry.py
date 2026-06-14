import cv2
import numpy as np


# Geometria de planos y compositing.
# Reutiliza la idea de order_points / four_point_transform de practica_homografia,
# pero ademas proyecta sprites con canal alfa y genera sombras sinteticas.


def order_points(pts):
    """Ordena 4 puntos como: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left  -> menor suma
    rect[2] = pts[np.argmax(s)]      # bottom-right -> mayor suma
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right -> menor diferencia
    rect[3] = pts[np.argmax(diff)]   # bottom-left -> mayor diferencia
    return rect


def alpha_composite(background, overlay_bgr, overlay_alpha):
    """Mezcla overlay (BGR) sobre background usando una mascara alfa float [0,1]."""
    alpha = overlay_alpha[:, :, np.newaxis]
    blended = overlay_bgr.astype(np.float32) * alpha + \
        background.astype(np.float32) * (1.0 - alpha)
    return blended.astype(np.uint8)


def warp_sprite_to_quad(sprite_rgba, dst_quad, out_size):
    """Proyecta un sprite RGBA a un cuadrilatero destino sobre un lienzo out_size (h, w).

    Devuelve (bgr, alpha_float) ya del tamano del lienzo, listos para componer.
    El sprite se mapea desde sus 4 esquinas (en orden TL, TR, BR, BL) al cuadrilatero.
    """
    h, w = sprite_rgba.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst_quad.astype('float32'))

    out_h, out_w = out_size
    bgr = cv2.warpPerspective(sprite_rgba[:, :, :3], M, (out_w, out_h))
    alpha = cv2.warpPerspective(sprite_rgba[:, :, 3], M, (out_w, out_h))
    return bgr, (alpha.astype(np.float32) / 255.0)


def standing_quad_from_floor(floor_quad, height_ratio):
    """A partir del cuadrilatero del piso (TL, TR, BR, BL) levanta un plano vertical.

    El mueble se apoya en la arista trasera del piso (TL-TR) y "se para" hacia arriba.
    height_ratio escala la altura respecto a la profundidad del cuadrilatero.
    Devuelve el cuadrilatero destino para el sprite (TL, TR, BR, BL del plano vertical).
    """
    rect = order_points(floor_quad)
    tl, tr, br, bl = rect
    # profundidad media del piso (cuanto se aleja) para escalar la altura
    depth = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    up = -np.array([0.0, depth * height_ratio])  # hacia arriba en pantalla (y negativo)
    # base del mueble = arista trasera del piso (la mas lejana, TL-TR)
    base_left, base_right = tl, tr
    top_left = base_left + up
    top_right = base_right + up
    # orden TL, TR, BR, BL para el sprite parado
    return np.array([top_left, top_right, base_right, base_left], dtype='float32')


def estimate_light_direction(image):
    """Estima una direccion de luz 2D simple a partir del gradiente de brillo.

    Idea: el lado mas iluminado de la imagen indica de donde viene la luz, asi
    la sombra cae hacia el lado opuesto. Devuelve un vector unitario (dx, dy).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    gx, gy = np.meshgrid(xs, ys)
    total = gray.sum() + 1e-6
    # centroide de brillo (donde la imagen es mas clara)
    cx = float((gray * gx).sum() / total)
    cy = float((gray * gy).sum() / total)
    light = np.array([cx, cy], dtype=np.float32)
    norm = np.linalg.norm(light)
    if norm < 1e-3:
        return np.array([0.4, 0.3], dtype=np.float32)  # default: luz desde arriba-izq
    return light / norm


def project_shadow(background, floor_quad, light_dir, opacity, blur=21):
    """Dibuja una sombra elipsoidal en el piso, desplazada segun la luz."""
    rect = order_points(floor_quad)
    center = rect.mean(axis=0)
    depth = np.linalg.norm(rect[3] - rect[0])
    # la sombra cae en sentido opuesto a la luz
    offset = -light_dir * depth * 0.4
    shadow_center = (center + offset).astype(int)
    axes = (int(np.linalg.norm(rect[1] - rect[0]) / 2),
            int(depth / 3))
    axes = (max(axes[0], 5), max(axes[1], 5))

    shadow = np.zeros(background.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow, tuple(shadow_center), axes, 0, 0, 360, 255, -1)
    shadow = cv2.GaussianBlur(shadow, (blur | 1, blur | 1), 0)
    alpha = (shadow.astype(np.float32) / 255.0) * opacity
    dark = np.zeros_like(background)
    return alpha_composite(background, dark, alpha)
