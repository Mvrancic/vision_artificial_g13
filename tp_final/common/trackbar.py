import cv2


# Helpers de trackbars, mismo estilo que tp_deteccion para mantener consistencia.

def create_trackbar(trackbar_name, window_name, slider_max, initial=0):
    cv2.createTrackbar(trackbar_name, window_name, initial, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))
