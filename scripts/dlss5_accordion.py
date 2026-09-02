import logging
import gradio as gr
from PIL import Image
import torch

from modules import scripts, devices, images, shared
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion

import sys
from pathlib import Path

# Add extension root to path
ext_dir = Path(__file__).resolve().parent.parent
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))

from dlss5_core.worker_manager import dlss5_manager
from dlss5_core.hardware_validator import check_hardware_compatibility

logger = logging.getLogger("dlss5.accordion")


class DLSS5IntegratedAccordion(scripts.Script):
    """
    DLSS 5 Neural Video & Image Enhancer Integrated Accordion.
    Appears directly inside txt2img and img2img tabs alongside built-in Forge accordions.
    """
    sorting_priority = 55  # Positioned cleanly alongside ControlNet & PiD

    def __init__(self):
        super().__init__()
        self.hw_compatible, self.hw_info = check_hardware_compatibility()

    def title(self):
        return "DLSS 5 Neural Enhancer Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        elem_prefix = "img2img" if is_img2img else "txt2img"

        with InputAccordion(False, label=self.title(), elem_id=f"{elem_prefix}_dlss5_accordion") as enable:
            if not self.hw_compatible:
                gr.Markdown(f"⚠️ *Note: {self.hw_info}. Standard Lanczos fallback will be used if enabled.*")

            with gr.Row():
                scale_mode = gr.Dropdown(
                    label="Upscaling Mode",
                    choices=[
                        "DLAA (1.0x)",
                        "Quality (1.5x)",
                        "Balanced (1.72x)",
                        "Performance (2.0x)",
                        "Ultra Performance (3.0x)",
                    ],
                    value="Quality (1.5x)",
                    elem_id=f"{elem_prefix}_dlss5_scale_mode",
                )
                model_preset = gr.Dropdown(
                    label="DLSS Model Preset",
                    choices=["Default", "J", "K", "L", "M"],
                    value="Default",
                    elem_id=f"{elem_prefix}_dlss5_model_preset",
                )
                nr_style = gr.Dropdown(
                    label="NR Style",
                    choices=["Default", "Natural", "Cinematic"],
                    value="Default",
                    elem_id=f"{elem_prefix}_dlss5_nr_style",
                )

            with gr.Row():
                nr_intensity = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="NR Intensity",
                    info="Global neural enhancement strength",
                    elem_id=f"{elem_prefix}_dlss5_intensity",
                )
                tone_strength = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Local Tone Strength",
                    info="Local contrast & tone boost",
                    elem_id=f"{elem_prefix}_dlss5_tone",
                )

            with gr.Row():
                struct_strength = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Local Structure Strength",
                    info="High-frequency edge & detail sharpening",
                    elem_id=f"{elem_prefix}_dlss5_structure",
                )
                skin_strength = gr.Slider(
                    minimum=-1.0,
                    maximum=2.0,
                    value=-1.0,
                    step=0.05,
                    label="Skin Structure Strength",
                    info="-1.0 = native auto; smoothing & texture tuning",
                    elem_id=f"{elem_prefix}_dlss5_skin",
                )

            with gr.Row():
                auto_mask = gr.Checkbox(
                    value=False,
                    label="Auto Semantic Mask",
                    info="Experimental semantic guidance mask",
                    elem_id=f"{elem_prefix}_dlss5_auto_mask",
                )

        self.infotext_fields = [
            (enable, "dlss5_enabled"),
            (scale_mode, "dlss5_scale_mode"),
            (model_preset, "dlss5_model_preset"),
            (nr_style, "dlss5_nr_style"),
            (nr_intensity, "dlss5_intensity"),
            (tone_strength, "dlss5_tone"),
            (struct_strength, "dlss5_structure"),
            (skin_strength, "dlss5_skin"),
            (auto_mask, "dlss5_auto_mask"),
        ]

        return [
            enable,
            scale_mode,
            model_preset,
            nr_style,
            nr_intensity,
            tone_strength,
            struct_strength,
            skin_strength,
            auto_mask,
        ]

    def postprocess_image_after_composite(
        self,
        p: StableDiffusionProcessing,
        pp: scripts.PostprocessImageArgs,
        enable: bool,
        scale_mode: str,
        model_preset: str,
        nr_style: str,
        nr_intensity: float,
        tone_strength: float,
        struct_strength: float,
        skin_strength: float,
        auto_mask: bool,
    ):
        """
        Instant forwarding hook: immediately upon completion of diffusion VAE decode,
        forwards the decoded image to the DLSS 5 D3D12 worker pipeline before gallery render.
        """
        if not enable:
            return

        try:
            logger.info(f"Forwarding image to DLSS 5 Engine: mode={scale_mode}, preset={model_preset}, intensity={nr_intensity}")
            enhanced = dlss5_manager.enhance_pil_image(
                image=pp.image,
                scale_mode=scale_mode,
                model_preset=model_preset,
                nr_intensity=nr_intensity,
                tone_strength=tone_strength,
                struct_strength=struct_strength,
                skin_strength=skin_strength,
                nr_style=nr_style,
                auto_mask=auto_mask,
            )

            pp.image = enhanced

            # Record parameters into PNG Info / generation params
            p.extra_generation_params.update({
                "dlss5_enabled": True,
                "dlss5_scale_mode": scale_mode,
                "dlss5_model_preset": model_preset,
                "dlss5_nr_style": nr_style,
                "dlss5_intensity": nr_intensity,
                "dlss5_tone": tone_strength,
                "dlss5_structure": struct_strength,
                "dlss5_skin": skin_strength,
                "dlss5_auto_mask": auto_mask,
            })

        except Exception as e:
            logger.error(f"Error during DLSS 5 image postprocessing: {e}", exc_info=True)

    def postprocess_batch_list(
        self,
        p: StableDiffusionProcessing,
        batch_params: scripts.PostprocessBatchListArgs,
        batch_number: int,
        enable: bool,
        scale_mode: str,
        model_preset: str,
        nr_style: str,
        nr_intensity: float,
        tone_strength: float,
        struct_strength: float,
        skin_strength: float,
        auto_mask: bool,
    ):
        """
        Handles Forge Neo native video models (SVD, Wan2.1, CogVideoX, AnimateDiff)
        by streaming temporal consecutive frames through DIS Optical Flow + DLSS 5 worker.
        """
        if not enable or not getattr(p, "is_video", False) and not getattr(p, "_is_video", False):
            return

        if not batch_params.images:
            return

        try:
            import numpy as np

            logger.info(f"Processing {len(batch_params.images)} video frames with DLSS 5 Temporal Pipeline...")
            raw_frames = []
            for img in batch_params.images:
                if isinstance(img, torch.Tensor):
                    arr = img.mul(255.0).clamp_(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                elif isinstance(img, Image.Image):
                    arr = np.array(img)
                else:
                    arr = np.array(img, dtype=np.uint8)
                raw_frames.append(arr)

            enhanced_frames = dlss5_manager.enhance_frame_sequence(
                frames=raw_frames,
                scale_mode=scale_mode,
                model_preset=model_preset,
                nr_intensity=nr_intensity,
                tone_strength=tone_strength,
                struct_strength=struct_strength,
                skin_strength=skin_strength,
                nr_style=nr_style,
                auto_mask=auto_mask,
            )

            # Replace batch frames in-place
            new_images = []
            for ef in enhanced_frames:
                new_images.append(Image.fromarray(ef))
            batch_params.images = new_images

        except Exception as e:
            logger.error(f"Error during DLSS 5 video batch processing: {e}", exc_info=True)
