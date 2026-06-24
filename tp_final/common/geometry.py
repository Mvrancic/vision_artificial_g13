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


def transform_quad(quad, offset=(0.0, 0.0), rotation_deg=0.0, scale=(1.0, 1.0)):
    """Traslada, rota y escala (ancho/alto por separado) un cuadrilatero alrededor de su centro.

    Se usa para "mover", "girar" y "agrandar/achicar" el mueble ya posicionado,
    sin tener que volver a marcar los 4 puntos del piso. El escalado se aplica en
    los ejes locales del objeto (ancho = TL-TR, alto = TL-BL) antes de rotar, asi
    "ancho" y "alto" siguen significando lo mismo aunque despues se gire el mueble.
    """
    scale_x, scale_y = scale
    center = quad.mean(axis=0)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype='float32')
    shifted = quad - center
    shifted[:, 0] *= float(scale_x)
    shifted[:, 1] *= float(scale_y)
    rotated = shifted @ rot.T
    moved = rotated + center + np.array(offset, dtype='float32')
    return moved.astype('float32')


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


def project_shadow(background, floor_quad, light_dir, opacity, offset=(0.0, 0.0), rotation_deg=0.0, scale=1.0, blur=35):
    """Dibuja una sombra de contacto (mas oscura cerca de la base, se disuelve hacia afuera).

    En vez de una elipse plana de opacidad uniforme, aplica una caida (gamma) sobre el
    blur para que el centro sea mas denso y los bordes se desvanezcan de forma gradual,
    lo que da una sensacion de "apoyado en el piso" en vez de un blob generico.
    offset/rotation_deg/scale permiten que la sombra siga al mueble cuando se lo mueve,
    rota o agranda/achica.
    """
    rect = order_points(floor_quad)
    center = rect.mean(axis=0) + np.array(offset, dtype='float32')
    depth = np.linalg.norm(rect[3] - rect[0])
    width = np.linalg.norm(rect[1] - rect[0])
    # la sombra cae en sentido opuesto a la luz
    light_offset = -light_dir * depth * 0.3
    shadow_center = center + light_offset
    axes = (max(int(width * 0.45 * scale), 5), max(int(depth * 0.24 * scale), 5))

    shadow = np.zeros(background.shape[:2], dtype=np.float32)
    cv2.ellipse(shadow, tuple(shadow_center.astype(int)), axes, rotation_deg, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (blur | 1, blur | 1), 0)
    shadow = np.clip(shadow, 0.0, 1.0)
    shadow = shadow ** 1.6  # nucleo mas denso, bordes mas suaves (look "contact shadow")
    alpha = shadow * opacity
    dark = np.zeros_like(background)
    return alpha_composite(background, dark, alpha)
