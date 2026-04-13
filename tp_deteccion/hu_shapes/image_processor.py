import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ProcessorParams:
    threshold: int
    invert: bool
    blur_size: int
    morph_radius: int
    min_area: int
    max_area: int


class ImageProcessor:
    """
    Shared pipeline for:
    - grayscale
    - threshold (+ optional invert)
    - morphological open/close
    - contour detection + area filtering
    - Hu moments (log-transformed)
    """

    def __init__(self, window_name: str = "TP - Hu Shapes"):
        self.window_name = window_name

        self._trackbars: Dict[str, int] = {
            "Threshold": 255,
            "Invert": 1,
            "Blur": 15,
            "Morph radius": 25,
            "Min area": 50000,
            "Max area": 500000,
        }

    def setup_ui(self) -> None:
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar("Threshold", self.window_name, 120, self._trackbars["Threshold"], lambda v: None)
        cv2.createTrackbar("Invert", self.window_name, 1, self._trackbars["Invert"], lambda v: None)
        cv2.createTrackbar("Blur", self.window_name, 3, self._trackbars["Blur"], lambda v: None)
        cv2.createTrackbar("Morph radius", self.window_name, 3, self._trackbars["Morph radius"], lambda v: None)
        cv2.createTrackbar("Min area", self.window_name, 3000, self._trackbars["Min area"], lambda v: None)
        cv2.createTrackbar("Max area", self.window_name, 250000, self._trackbars["Max area"], lambda v: None)

    def get_params(self) -> ProcessorParams:
        thr = int(cv2.getTrackbarPos("Threshold", self.window_name))
        invert = int(cv2.getTrackbarPos("Invert", self.window_name)) == 1
        blur_size = int(cv2.getTrackbarPos("Blur", self.window_name))
        morph_radius = int(cv2.getTrackbarPos("Morph radius", self.window_name))
        min_area = int(cv2.getTrackbarPos("Min area", self.window_name))
        max_area = int(cv2.getTrackbarPos("Max area", self.window_name))

        if blur_size > 0 and blur_size % 2 == 0:
            blur_size += 1
        morph_radius = max(0, morph_radius)
        if max_area < min_area:
            max_area = min_area

        return ProcessorParams(
            threshold=thr,
            invert=invert,
            blur_size=blur_size,
            morph_radius=morph_radius,
            min_area=min_area,
            max_area=max_area,
        )

    def preprocess(self, frame_bgr: np.ndarray, params: ProcessorParams) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if params.blur_size > 1:
            gray = cv2.GaussianBlur(gray, (params.blur_size, params.blur_size), 0)
        _, bin_img = cv2.threshold(gray, params.threshold, 255, cv2.THRESH_BINARY)

        if params.invert:
            bin_img = 255 - bin_img

        if params.morph_radius > 0:
            k = params.morph_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        return bin_img

    def find_contours(self, bin_img: np.ndarray, params: ProcessorParams) -> List[np.ndarray]:
        contours, _hier = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered: List[np.ndarray] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if params.min_area <= area <= params.max_area:
                filtered.append(contour)
        return filtered

    @staticmethod
    def hu_log_transform(hu: np.ndarray) -> np.ndarray:
        out = np.zeros((7,), dtype=np.float32)
        hu = hu.reshape(-1)
        for index in range(7):
            value = float(hu[index])
            if value == 0.0:
                out[index] = 0.0
            else:
                out[index] = float(-math.copysign(1.0, value) * math.log10(abs(value)))
        return out

    def contour_to_hu(self, contour: np.ndarray) -> np.ndarray:
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments)
        return self.hu_log_transform(hu)

    @staticmethod
    def pick_primary_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def draw_contour_label(
        frame_bgr: np.ndarray,
        contour: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        cv2.drawContours(frame_bgr, [contour], -1, color, thickness)
        x, y, _w, _h = cv2.boundingRect(contour)
        cv2.putText(frame_bgr, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

