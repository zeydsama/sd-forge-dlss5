import logging
import cv2
import numpy as np

logger = logging.getLogger("dlss5.optical_flow")


class OpticalFlowGenerator:
    """Computes dense optical flow (motion vectors) in RG float16 format for DLSS temporal accumulation."""

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        self.dis = cv2.DISOpticalFlow_create(preset)
        self.dis.setFinestScale(1)
        self.dis.setGradientDescentIterations(16)
        self.prev_gray: np.ndarray | None = None

    def reset(self):
        """Reset historical frame cache."""
        self.prev_gray = None

    def compute_motion_vectors(self, frame_rgba: np.ndarray) -> bytes:
        """
        Compute dense optical flow between previous and current frame.
        Input: RGBA8 numpy array [H, W, 4]
        Output: Packed float16 binary bytes for (dx, dy) motion vectors [H, W, 2] in pixels.
        """
        h, w = frame_rgba.shape[:2]

        # Convert RGBA to single-channel Grayscale for DIS Optical Flow
        curr_gray = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != curr_gray.shape:
            # First frame or resolution change: zero motion vectors
            self.prev_gray = curr_gray
            flow = np.zeros((h, w, 2), dtype=np.float16)
            return flow.tobytes()

        # Calculate flow (returns float32 array of shape [H, W, 2])
        flow_f32 = self.dis.calc(self.prev_gray, curr_gray, None)
        self.prev_gray = curr_gray

        # Cast to float16 as required by DLSS worker protocol v4
        flow_f16 = flow_f32.astype(np.float16)
        return flow_f16.tobytes()

    @staticmethod
    def create_zero_motion_vectors(width: int, height: int) -> bytes:
        """Create a zero motion vector buffer for still images."""
        return np.zeros((height, width, 2), dtype=np.float16).tobytes()
