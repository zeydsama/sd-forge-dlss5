import cv2
import numpy as np


class SceneDetector:
    """Detects abrupt scene cuts between frames to signal DLSS history resets."""

    def __init__(self, threshold: float = 0.24):
        self.threshold = threshold
        self.prev_hist: np.ndarray | None = None

    def reset(self):
        self.prev_hist = None

    def is_scene_cut(self, frame_rgba: np.ndarray) -> bool:
        """
        Calculates Bhattacharyya distance between consecutive frame HSV histograms.
        Returns True if a scene transition is detected.
        """
        hsv = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)

        # Compute 2D Hue-Saturation histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if self.prev_hist is None:
            self.prev_hist = hist
            return True  # First frame is always considered a clean reset

        # Compare histograms using Bhattacharyya distance (0 = identical, 1 = completely different)
        distance = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist

        return bool(distance > self.threshold)
