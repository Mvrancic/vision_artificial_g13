import cv2
import numpy as np


WINDOW_MAIN = "TP4 - Webcam"
WINDOW_WARPED = "TP4 - Vista frontal"
WARP_SIZE = 300
GRID_CELLS = 3

MODE_VIEW = "visualizacion"
MODE_QR = "qr"
MODE_MANUAL = "manual"


def order_points(points):
    pts = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def build_destination_square(size):
    edge = float(size - 1)
    return np.array(
        [[0.0, 0.0], [edge, 0.0], [edge, edge], [0.0, edge]],
        dtype=np.float32,
    )


class HomographyApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la webcam.")

        self.detector = cv2.QRCodeDetector()
        self.mode = MODE_VIEW
        self.manual_points = []
        self.last_qr_points = None
        self.homography = None
        self.inverse_homography = None
        self.destination_square = build_destination_square(WARP_SIZE)

        cv2.namedWindow(WINDOW_MAIN)
        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if self.mode != MODE_MANUAL or event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.manual_points) >= 4:
            return

        self.manual_points.append((float(x), float(y)))
        if len(self.manual_points) == 4:
            self.update_homography(self.manual_points)
            self.mode = MODE_VIEW
            self.manual_points = []

    def update_homography(self, source_points):
        ordered_points = order_points(source_points)
        homography = cv2.getPerspectiveTransform(
            ordered_points, self.destination_square
        )
        inverse = cv2.getPerspectiveTransform(self.destination_square, ordered_points)

        self.homography = homography
        self.inverse_homography = inverse

    def detect_qr_points(self, frame):
        found, points = self.detector.detect(frame)
        if not found or points is None:
            return None

        corners = points.reshape(-1, 2)
        if corners.shape[0] != 4:
            return None
        return order_points(corners)

    def draw_overlay_text(self, frame):
        mode_label = {
            MODE_VIEW: "Modo: visualizacion",
            MODE_QR: "Modo: QR",
            MODE_MANUAL: "Modo: homografia asistida",
        }[self.mode]

        lines = [mode_label, "Teclas: q=QR  h=manual  Esc=salir"]

        if self.mode == MODE_MANUAL:
            lines.append(f"Puntos: {len(self.manual_points)}/4")
            lines.append("Orden: top-left, top-right, bottom-right, bottom-left")
            lines.append("Cualquier tecla aborta")
        elif self.mode == MODE_QR:
            status = "QR detectado" if self.last_qr_points is not None else "Buscando QR"
            lines.append(status)
            lines.append("Presiona cualquier tecla para confirmar")

        y = 28
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 28

    def draw_manual_points(self, frame):
        for index, point in enumerate(self.manual_points, start=1):
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(index),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(self.manual_points) >= 2:
            polyline = np.array(self.manual_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [polyline], False, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_qr_preview(self, frame):
        if self.last_qr_points is None:
            return

        polygon = self.last_qr_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_grid(self, frame):
        if self.inverse_homography is None:
            return

        grid_color = (255, 120, 0)
        last = WARP_SIZE - 1

        for idx in range(GRID_CELLS + 1):
            coord = idx * last / GRID_CELLS

            vertical = np.array(
                [[[coord, 0.0]], [[coord, last]]],
                dtype=np.float32,
            )
            horizontal = np.array(
                [[[0.0, coord]], [[last, coord]]],
                dtype=np.float32,
            )

            projected_vertical = cv2.perspectiveTransform(
                vertical, self.inverse_homography
            )
            projected_horizontal = cv2.perspectiveTransform(
                horizontal, self.inverse_homography
            )

            cv2.polylines(
                frame,
                [np.round(projected_vertical).astype(np.int32)],
                False,
                grid_color,
                2,
                cv2.LINE_AA,
            )
            cv2.polylines(
                frame,
                [np.round(projected_horizontal).astype(np.int32)],
                False,
                grid_color,
                2,
                cv2.LINE_AA,
            )

    def show_warped_view(self, frame):
        if self.homography is None:
            try:
                cv2.destroyWindow(WINDOW_WARPED)
            except cv2.error:
                pass
            return

        warped = cv2.warpPerspective(frame, self.homography, (WARP_SIZE, WARP_SIZE))
        cv2.imshow(WINDOW_WARPED, warped)

    def handle_key(self, key):
        if key == 27:
            return False

        if self.mode == MODE_VIEW:
            if key == ord("q"):
                self.mode = MODE_QR
                self.last_qr_points = None
            elif key == ord("h"):
                self.mode = MODE_MANUAL
                self.manual_points = []
            return True

        if self.mode == MODE_QR and key != -1:
            if self.last_qr_points is not None:
                self.update_homography(self.last_qr_points)
            self.mode = MODE_VIEW
            self.last_qr_points = None
            return True

        if self.mode == MODE_MANUAL and key != -1 and len(self.manual_points) < 4:
            self.manual_points = []
            self.mode = MODE_VIEW
            return True

        return True

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            display = frame.copy()

            if self.mode == MODE_QR:
                self.last_qr_points = self.detect_qr_points(frame)
                self.draw_qr_preview(display)

            if self.mode == MODE_MANUAL:
                self.draw_manual_points(display)

            if self.homography is not None:
                self.draw_grid(display)
                self.show_warped_view(frame)

            self.draw_overlay_text(display)
            cv2.imshow(WINDOW_MAIN, display)

            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key(key if key != 255 else -1):
                break

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = HomographyApp()
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
