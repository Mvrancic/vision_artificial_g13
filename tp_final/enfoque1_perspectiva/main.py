import os
import sys
import cv2
import numpy as np

# permitir importar el paquete common cuando se corre como script suelto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import assets, geometry, sources
from common.point_picker import PointPicker
from common.trackbar import create_trackbar, get_trackbar_value

WINDOW = 'Enfoque 1 - Visualizacion de muebles en perspectiva real'


def render(background, floor_quad, sprite, height_ratio, shadow_opacity, show_shadow):
    """Compone el mueble parado sobre el cuadrilatero del piso + sombra."""
    out = background.copy()
    h, w = out.shape[:2]

    if show_shadow and shadow_opacity > 0:
        light = geometry.estimate_light_direction(background)
        out = geometry.project_shadow(out, floor_quad, light, shadow_opacity)

    # el mueble se "para" sobre la arista trasera del piso marcado
    standing = geometry.standing_quad_from_floor(floor_quad, height_ratio)
    bgr, alpha = geometry.warp_sprite_to_quad(sprite, standing, (h, w))
    out = geometry.alpha_composite(out, bgr, alpha)
    return out


def draw_help(image, furniture_name):
    lines = [
        'Mueble: %s  (n=siguiente)' % furniture_name,
        'Trackbars: altura / sombra',
        's=guardar  r=re-marcar piso  q=salir',
    ]
    y = 25
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 28
    return image


def run(image_path=None, use_camera=False):
    # 1) obtener la foto de la habitacion
    if use_camera:
        background = sources.grab_frame_from_camera()
        if background is None:
            print('Captura cancelada.')
            return
    else:
        if image_path is None:
            image_path = os.path.join(sources.REPO_IMAGES, 'pool2.png')
        background = sources.read_image(image_path)
    background = sources.resize_to_width(background, 900)

    furniture = assets.load_furniture()
    idx = 0

    # 2) marcar 4 puntos en el piso
    floor_quad = PointPicker(background, 4, 'Marca 4 puntos en el piso').run()
    if floor_quad is None:
        print('No se marcaron los 4 puntos.')
        return
    floor_quad = geometry.order_points(floor_quad)

    # 3) ventana principal con controles
    cv2.namedWindow(WINDOW)
    create_trackbar('Altura x100', WINDOW, 200, initial=80)
    create_trackbar('Sombra x100', WINDOW, 100, initial=45)
    show_shadow = True
    save_count = 0

    while True:
        height_ratio = max(get_trackbar_value('Altura x100', WINDOW), 1) / 100.0
        shadow_opacity = get_trackbar_value('Sombra x100', WINDOW) / 100.0
        name, sprite = furniture[idx]

        frame = render(background, floor_quad, sprite, height_ratio,
                       shadow_opacity, show_shadow)
        draw_help(frame, name)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(furniture)
        elif key == ord('t'):
            show_shadow = not show_shadow
        elif key == ord('r'):
            new_quad = PointPicker(background, 4, 'Marca 4 puntos en el piso').run()
            if new_quad is not None:
                floor_quad = geometry.order_points(new_quad)
        elif key == ord('s'):
            out_path = os.path.join(os.path.dirname(__file__),
                                    'resultado_%d.png' % save_count)
            cv2.imwrite(out_path, frame)
            print('Guardado:', out_path)
            save_count += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
