from PIL import Image
from modules import shared, upscaler
import sys
from pathlib import Path

# Add extension root to path
ext_dir = Path(__file__).resolve().parent.parent
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))

from dlss5_core.worker_manager import dlss5_manager


class UpscalerDLSS5(upscaler.Upscaler):
    """Exposes DLSS 5 into standard Forge upscaler dropdowns (e.g. Hires. Fix, Extras)."""

    def __init__(self, user_path=None):
        self.name = "DLSS 5"
        self.scalers = []
        super().__init__()

        # Register standard model modes
        modes = [
            ("DLSS 5 (DLAA 1.0x)", "DLAA (1.0x)", 1.0),
            ("DLSS 5 (Quality 1.5x)", "Quality (1.5x)", 1.5),
            ("DLSS 5 (Balanced 1.7x)", "Balanced (1.72x)", 1.724),
            ("DLSS 5 (Performance 2.0x)", "Performance (2.0x)", 2.0),
            ("DLSS 5 (Ultra Performance 3.0x)", "Ultra Performance (3.0x)", 3.0),
        ]

        for display_name, scale_mode, scale_val in modes:
            data = upscaler.UpscalerData(
                name=display_name,
                path=None,
                upscaler=self,
                scale=scale_val,
                model_name=scale_mode,
            )
            self.scalers.append(data)

    def do_upscale(self, img: Image.Image, selected_model: str = "Quality (1.5x)") -> Image.Image:
        scale_mode = selected_model if selected_model in dlss5_manager.calculate_dimensions.__code__.co_varnames or "x" in selected_model else "Quality (1.5x)"
        return dlss5_manager.enhance_pil_image(img, scale_mode=scale_mode)


# Register upscaler instances into shared.sd_upscalers if initialized
try:
    dlss5_upscaler_instance = UpscalerDLSS5()
    for u in dlss5_upscaler_instance.scalers:
        if not any(existing.name == u.name for existing in shared.sd_upscalers):
            shared.sd_upscalers.append(u)
except Exception:
    pass
