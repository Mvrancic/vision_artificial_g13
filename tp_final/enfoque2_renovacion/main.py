import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import sources
from common.trackbar import create_trackbar, get_trackbar_value

WINDOW = 'Enfoque 2 - Renovacion inteligente (borrar + inpainting)'


def segment_object(image, rect):
    """Refina la seleccion del usuario con grabCut y devuelve la mascara del mueble.

    rect = (x, y, w, h). Reutiliza la idea de practica_segmentacion/grabcut.py.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
    # 1 y 3 = foreground (seguro / probable)
    object_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return object_mask.astype(np.uint8)


def remove_and_fill(image, object_mask, radius, method, dilate_iter):
    """Borra el objeto y reconstruye el fondo con inpainting."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(object_mask, kernel, iterations=dilate_iter)
    flag = cv2.INPAINT_TELEA if method == 0 else cv2.INPAINT_NS
    return cv2.inpaint(image, mask, max(radius, 1), flag), mask


def draw_help(image, method, radius, dilate_iter):
    method_name = 'TELEA' if method == 0 else 'NS'
    lines = [
        'Metodo: %s (m=cambiar)  radio=%d  dilatacion=%d' % (method_name, radius, dilate_iter),
        'r=re-seleccionar mueble   s=guardar   q=salir',
    ]
    y = 25
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 28
    return image


def select_and_segment(image):
    """Pide la ROI del mueble y devuelve su mascara (o None si se cancela)."""
    rect = cv2.selectROI('Selecciona el mueble a quitar (ENTER=ok)', image,
                          fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Selecciona el mueble a quitar (ENTER=ok)')
    if rect[2] == 0 or rect[3] == 0:
        return None
    return segment_object(image, rect)


def run(image_path=None, use_camera=False):
    if use_camera:
        image = sources.grab_frame_from_camera()
        if image is None:
            print('Captura cancelada.')
            return
    else:
        if image_path is None:
            image_path = os.path.join(sources.REPO_IMAGES, 'messi.jpg')
        image = sources.read_image(image_path)
    image = sources.resize_to_width(image, 900)

    object_mask = select_and_segment(image)
    if object_mask is None:
        print('No se selecciono ningun objeto.')
        return

    cv2.namedWindow(WINDOW)
    create_trackbar('Radio inpaint', WINDOW, 20, initial=5)
    create_trackbar('Dilatacion', WINDOW, 15, initial=4)
    method = 0
    save_count = 0

    while True:
        radius = get_trackbar_value('Radio inpaint', WINDOW)
        dilate_iter = get_trackbar_value('Dilatacion', WINDOW)
        result, mask = remove_and_fill(image, object_mask, radius, method, dilate_iter)

        # vista comparativa: original con mascara resaltada | resultado limpio
        marked = image.copy()
        marked[mask > 0] = (0, 0, 255)
        marked = cv2.addWeighted(image, 0.6, marked, 0.4, 0)
        combined = np.hstack([marked, result])
        draw_help(combined, method, radius, dilate_iter)
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            method = 1 - method
        elif key == ord('r'):
            new_mask = select_and_segment(image)
            if new_mask is not None:
                object_mask = new_mask
        elif key == ord('s'):
            out_path = os.path.join(os.path.dirname(__file__),
                                    'limpio_%d.png' % save_count)
            cv2.imwrite(out_path, result)
            print('Guardado:', out_path)
            save_count += 1

    cv2.destroyAllWindows()
    return result


if __name__ == '__main__':
    run()
