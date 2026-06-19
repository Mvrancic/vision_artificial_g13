import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import sources
from common.trackbar import create_trackbar, get_trackbar_value
from enfoque2_renovacion.backends import clean_mask, inpaint_lama, inpaint_opencv, segment_object as segment_object_backend

WINDOW = 'Enfoque 2 - Renovacion inteligente (borrar + inpainting)'


def segment_object(image, rect, mode="auto"):
    """Segmenta el objeto dentro de la ROI usando el backend indicado."""
    return segment_object_backend(image, rect, mode=mode)


def remove_and_fill(image, object_mask, radius, method, dilate_iter, expand_px=0, feather_px=0):
    """Borra el objeto y reconstruye el fondo con inpainting.

    method:
      0 -> TELEA
      1 -> Navier-Stokes
      2 -> LaMa
    """
    mask = clean_mask(
        object_mask,
        dilate_iter=dilate_iter,
        expand_px=expand_px,
        feather_px=feather_px,
    )
    if method == 2:
        result = inpaint_lama(image, mask)
        return result.image, result.mask, result.backend_name
    result = inpaint_opencv(image, mask, radius=radius, method=method)
    return result.image, result.mask, result.backend_name


def draw_help(image, method, radius, dilate_iter):
    method_name = 'TELEA' if method == 0 else 'NS' if method == 1 else 'LaMa'
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

    segmented = select_and_segment(image)
    object_mask = segmented[0] if segmented is not None else None
    segment_name = segmented[1] if segmented is not None else "N/A"
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
        try:
            result, mask, backend_name = remove_and_fill(image, object_mask, radius, method, dilate_iter)
        except Exception as exc:
            result = image.copy()
            mask = clean_mask(object_mask, dilate_iter=dilate_iter)
            backend_name = f'Error: {exc}'

        # vista comparativa: original con mascara resaltada | resultado limpio
        marked = image.copy()
        marked[mask > 0] = (0, 0, 255)
        marked = cv2.addWeighted(image, 0.6, marked, 0.4, 0)
        combined = np.hstack([marked, result])
        draw_help(combined, method, radius, dilate_iter)
        cv2.putText(combined, backend_name, (10, combined.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(combined, backend_name, (10, combined.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(combined, f'Segmentacion: {segment_name}', (10, combined.shape[0] - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(combined, f'Segmentacion: {segment_name}', (10, combined.shape[0] - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            method = (method + 1) % 3
        elif key == ord('r'):
            segmented = select_and_segment(image)
            if segmented is not None:
                object_mask, segment_name = segmented
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
