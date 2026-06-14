import cv2
import numpy as np


# Selector de puntos con el mouse sobre una imagen fija.
# Se usa para marcar las 4 esquinas del piso donde va el mueble (enfoque 1).

COLOR_POINT = (0, 0, 255)
COLOR_LINE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)


class PointPicker:
    def __init__(self, image, n_points=4, window='Seleccionar puntos'):
        self.image = image
        self.n_points = n_points
        self.window = window
        self.points = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.n_points:
            self.points.append((x, y))

    def _draw(self):
        canvas = self.image.copy()
        for i, p in enumerate(self.points):
            cv2.circle(canvas, p, 6, COLOR_POINT, -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        if len(self.points) >= 2:
            cv2.polylines(canvas, [np.array(self.points, dtype=np.int32)],
                          len(self.points) == self.n_points, COLOR_LINE, 2)
        msg = 'Marca %d puntos (%d/%d) | r=reset  ENTER=confirmar  q=cancelar' % (
            self.n_points, len(self.points), self.n_points)
        cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
        return canvas

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)
        while True:
            cv2.imshow(self.window, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.points = []
            elif key == ord('q'):
                self.points = []
                break
            elif key in (13, 10) and len(self.points) == self.n_points:
                break
        cv2.destroyWindow(self.window)
        if len(self.points) == self.n_points:
            return np.array(self.points, dtype='float32')
        return None
