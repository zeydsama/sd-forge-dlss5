import unittest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add extension root to path
ext_dir = Path(__file__).resolve().parent.parent
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))

from dlss5_core.protocol_v4 import (
    pack_video_header,
    unpack_setup_response,
    pack_frame_header,
    unpack_out_header,
    VIDEO_MAGIC,
    SETUP_RESPONSE_MAGIC,
    FRAME_MAGIC,
    OUT_MAGIC,
)
from dlss5_core.optical_flow import OpticalFlowGenerator
from dlss5_core.scene_detector import SceneDetector
from dlss5_core.worker_manager import DLSS5Manager


class TestDLSS5Core(unittest.TestCase):

    def test_protocol_v4_header_packing(self):
        # 1. Video Setup Header Packing
        pkt = pack_video_header(
            input_width=512,
            input_height=512,
            output_width=768,
            output_height=768,
            warmup_frames=0,
            frame_count=1,
            perf_quality=2,  # Quality (1.5x)
            dlss_model_preset=10,  # Preset J
            intensity=1.25,
            local_tone=0.9,
            local_structure=1.5,
            skin_structure=-1.0,
        )
        self.assertEqual(len(pkt), 72)

        # 2. Frame Header Packing
        frame_pkt = pack_frame_header(frame_index=42, reset=1, flags=0, pts=1000)
        self.assertEqual(len(frame_pkt), 24)

    def test_optical_flow_generator(self):
        flow_gen = OpticalFlowGenerator()

        # Zero flow test
        frame1 = np.zeros((128, 128, 4), dtype=np.uint8)
        flow1 = flow_gen.compute_motion_vectors(frame1)
        self.assertEqual(len(flow1), 128 * 128 * 2 * 2)  # 2 channels * float16 (2 bytes)

        # Non-zero motion test (shift image)
        frame2 = np.zeros((128, 128, 4), dtype=np.uint8)
        frame2[10:60, 10:60] = 255
        frame3 = np.zeros((128, 128, 4), dtype=np.uint8)
        frame3[15:65, 15:65] = 255

        _ = flow_gen.compute_motion_vectors(frame2)
        flow_bytes = flow_gen.compute_motion_vectors(frame3)
        flow_arr = np.frombuffer(flow_bytes, dtype=np.float16).reshape((128, 128, 2))
        self.assertEqual(flow_arr.shape, (128, 128, 2))

    def test_scene_detector(self):
        detector = SceneDetector(threshold=0.24)

        # Frame 1 (first frame is reset)
        f1 = np.zeros((128, 128, 4), dtype=np.uint8)
        f1[:, :, 0] = 255  # Red
        self.assertTrue(detector.is_scene_cut(f1))

        # Frame 2 (identical)
        self.assertFalse(detector.is_scene_cut(f1))

        # Frame 3 (drastic cut to Blue)
        f2 = np.zeros((128, 128, 4), dtype=np.uint8)
        f2[:, :, 2] = 255  # Blue
        self.assertTrue(detector.is_scene_cut(f2))

    def test_manager_dimensions(self):
        mgr = DLSS5Manager()
        out_w, out_h = mgr.calculate_dimensions(512, 512, "Quality (1.5x)")
        self.assertEqual((out_w, out_h), (768, 768))

        out_w, out_h = mgr.calculate_dimensions(512, 512, "Performance (2.0x)")
        self.assertEqual((out_w, out_h), (1024, 1024))

        out_w, out_h = mgr.calculate_dimensions(512, 512, "DLAA (1.0x)")
        self.assertEqual((out_w, out_h), (512, 512))

    def test_fallback_enhancement(self):
        mgr = DLSS5Manager()
        img = Image.new("RGB", (256, 256), color=(255, 0, 0))
        enhanced = mgr.enhance_pil_image(img, scale_mode="Quality (1.5x)")
        self.assertEqual(enhanced.size, (384, 384))


if __name__ == "__main__":
    unittest.main()
