import os
import cv2


# Manejo de fuentes de imagen: webcam o archivos estaticos.
# Permite que los tres enfoques trabajen igual con una foto fija o con la camara,
# manteniendo la interfaz estilo tp_deteccion (ventana OpenCV).

# Ruta a las imagenes que ya tiene el repo, por si se quiere probar sin webcam.
REPO_IMAGES = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'images')


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la camara (index=%d)' % index)
    return cap


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError('No se pudo leer la imagen: %s' % path)
    return image


def grab_frame_from_camera(index=0):
    """Abre la camara, deja al usuario encuadrar y captura un frame con 'c'."""
    cap = open_camera(index)
    window = 'Capturar foto (c=capturar, q=cancelar)'
    cv2.namedWindow(window)
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            captured = frame.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyWindow(window)
    return captured


def resize_to_width(image, target_width=900):
    """Reescala manteniendo proporcion, para que entre comodo en pantalla."""
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / float(w)
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
