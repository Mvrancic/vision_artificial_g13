import os
import cv2
import numpy as np


# Genera sprites de muebles (PNG con canal alfa) de forma procedural, asi el TP
# funciona sin depender de descargas. Cada sprite es una "vista frontal" del mueble
# que luego se proyecta en perspectiva con homografia (enfoque 1).

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')


def _blank(w, h):
    return np.zeros((h, w, 4), dtype=np.uint8)


def _rounded_rect(img, p1, p2, color, radius=15):
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(img, (cx, cy), radius, color, -1)


def make_sofa():
    w, h = 420, 240
    img = _blank(w, h)
    body = (70, 90, 150, 255)       # bordo
    cushion = (90, 120, 190, 255)
    _rounded_rect(img, (20, 80), (400, 220), body, 25)      # base + respaldo
    _rounded_rect(img, (10, 70), (70, 200), body, 20)       # apoyabrazos izq
    _rounded_rect(img, (350, 70), (410, 200), body, 20)     # apoyabrazos der
    _rounded_rect(img, (80, 110), (200, 175), cushion, 12)  # almohadon 1
    _rounded_rect(img, (215, 110), (340, 175), cushion, 12)  # almohadon 2
    return img


def make_armchair():
    w, h = 240, 240
    img = _blank(w, h)
    body = (60, 75, 130, 255)
    cushion = (80, 105, 170, 255)
    _rounded_rect(img, (30, 70), (210, 220), body, 25)
    _rounded_rect(img, (15, 60), (65, 200), body, 18)
    _rounded_rect(img, (175, 60), (225, 200), body, 18)
    _rounded_rect(img, (70, 100), (170, 175), cushion, 12)
    return img


def make_table():
    w, h = 360, 200
    img = _blank(w, h)
    wood = (45, 90, 140, 255)
    dark = (30, 60, 95, 255)
    _rounded_rect(img, (20, 40), (340, 90), wood, 12)    # tablero
    cv2.rectangle(img, (40, 90), (60, 190), dark, -1)    # patas
    cv2.rectangle(img, (300, 90), (320, 190), dark, -1)
    return img


def make_bed():
    w, h = 460, 260
    img = _blank(w, h)
    frame = (70, 70, 70, 255)
    sheet = (200, 200, 210, 255)
    pillow = (240, 240, 245, 255)
    _rounded_rect(img, (20, 60), (440, 240), frame, 20)
    _rounded_rect(img, (35, 90), (425, 230), sheet, 14)
    _rounded_rect(img, (55, 100), (180, 150), pillow, 12)
    _rounded_rect(img, (200, 100), (325, 150), pillow, 12)
    return img


def make_plant():
    w, h = 180, 260
    img = _blank(w, h)
    pot = (40, 70, 110, 255)
    leaf = (60, 140, 60, 255)
    cv2.rectangle(img, (60, 190), (120, 250), pot, -1)
    cv2.circle(img, (90, 120), 70, leaf, -1)
    cv2.circle(img, (55, 90), 45, leaf, -1)
    cv2.circle(img, (125, 90), 45, leaf, -1)
    return img


FACTORIES = {
    'sofa': make_sofa,
    'armchair': make_armchair,
    'table': make_table,
    'bed': make_bed,
    'plant': make_plant,
}


def ensure_assets():
    """Genera los PNG en disco si no existen y devuelve {nombre: ruta}."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    paths = {}
    for name, factory in FACTORIES.items():
        path = os.path.join(ASSETS_DIR, name + '.png')
        if not os.path.exists(path):
            cv2.imwrite(path, factory())
        paths[name] = path
    return paths


def load_furniture():
    """Devuelve una lista [(nombre, sprite_rgba)] lista para usar."""
    ensure_assets()
    items = []
    for name, factory in FACTORIES.items():
        path = os.path.join(ASSETS_DIR, name + '.png')
        sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if sprite is None or sprite.shape[2] != 4:
            sprite = factory()
        items.append((name, sprite))
    return items


if __name__ == '__main__':
    print('Generando sprites en', os.path.abspath(ASSETS_DIR))
    for name, path in ensure_assets().items():
        print(' -', name, '->', path)
