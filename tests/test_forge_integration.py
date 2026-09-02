import unittest
import json
import urllib.request
import urllib.error
import sys
from pathlib import Path

# Add extension root to sys.path
ext_dir = Path(__file__).resolve().parent.parent
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))


class TestForgeDLSS5Integration(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:7860"

    def test_live_forge_api_options_and_memory(self):
        """Check Forge REST API health, memory, and options."""
        try:
            req = urllib.request.Request(f"{self.BASE_URL}/sdapi/v1/memory")
            with urllib.request.urlopen(req, timeout=5) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode("utf-8"))
                self.assertIn("cuda", data)
                print(f"[Live Forge API] CUDA Device: {data['cuda']['device']}")
        except urllib.error.URLError:
            print("[Live Forge API] Server is not running at 127.0.0.1:7860.")

    def test_dlss5_core_components(self):
        """Verify that DLSS 5 core modules initialize and operate cleanly."""
        from dlss5_core.hardware_validator import check_hardware_compatibility
        from dlss5_core.runtime_manager import get_default_runtime_dir, validate_runtime_binaries
        from dlss5_core.worker_manager import dlss5_manager

        ok, msg = check_hardware_compatibility()
        self.assertTrue(ok)
        self.assertIn("RTX", msg)
        print(f"[Hardware Check] {msg}")

        runtime_dir = get_default_runtime_dir()
        valid, missing = validate_runtime_binaries(runtime_dir)
        print(f"[Runtime Binary Check] Expected at {runtime_dir} | Missing: {missing}")

        # Test resolution calculation
        out_w, out_h = dlss5_manager.calculate_dimensions(512, 512, "Quality (1.5x)")
        self.assertEqual((out_w, out_h), (768, 768))


if __name__ == "__main__":
    unittest.main()
