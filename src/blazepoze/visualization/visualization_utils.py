"""Lightweight helpers for OpenCV visualizations."""

from typing import Tuple

import cv2 as cv
import numpy as np


class VisualizationUtils:
    """Utility wrapper for OpenCV display functions."""

    display_enabled: bool = True

    @staticmethod
    def overlay_text(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        color: Tuple[int, int, int] = (0, 255, 0),
        scale: float = 1.0,
        thickness: int = 2,
    ) -> None:
        """Draw text on an image in-place."""
        cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    @staticmethod
    def show_image(window_name: str, frame: np.ndarray) -> bool:
        """Display an image if enabled and return ``False`` when ``q`` is pressed."""
        if not VisualizationUtils.display_enabled:
            return True

        cv.imshow(window_name, frame)
        key = cv.waitKey(1) & 0xFF
        return key != ord("q")
