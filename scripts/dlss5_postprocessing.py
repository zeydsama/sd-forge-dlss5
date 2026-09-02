import gradio as gr
from PIL import Image

from modules import scripts_postprocessing, shared
import sys
from pathlib import Path

# Add extension root to path
ext_dir = Path(__file__).resolve().parent.parent
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))

from dlss5_core.worker_manager import dlss5_manager


class ScriptPostprocessingDLSS5(scripts_postprocessing.ScriptPostprocessing):
    name = "DLSS 5 Neural Enhancer"
    order = 950

    def ui(self):
        with gr.Accordion("DLSS 5 Neural Enhancer", open=False, elem_id="extras_dlss5_accordion"):
            enable = gr.Checkbox(label="Enable DLSS 5 Enhancement", value=False)
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
                )
                model_preset = gr.Dropdown(
                    label="DLSS Model Preset",
                    choices=["Default", "J", "K", "L", "M"],
                    value="Default",
                )
            with gr.Row():
                nr_intensity = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="NR Intensity")
                tone_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Local Tone Strength")
                struct_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Local Structure Strength")
                skin_strength = gr.Slider(-1.0, 2.0, value=-1.0, step=0.05, label="Skin Structure Strength")
            with gr.Row():
                nr_style = gr.Dropdown(choices=["Default", "Natural", "Cinematic"], value="Default", label="NR Style")
                auto_mask = gr.Checkbox(value=False, label="Auto Semantic Mask")

        return {
            "enable": enable,
            "scale_mode": scale_mode,
            "model_preset": model_preset,
            "nr_intensity": nr_intensity,
            "tone_strength": tone_strength,
            "struct_strength": struct_strength,
            "skin_strength": skin_strength,
            "nr_style": nr_style,
            "auto_mask": auto_mask,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        enable=False,
        scale_mode="Quality (1.5x)",
        model_preset="Default",
        nr_intensity=1.0,
        tone_strength=1.0,
        struct_strength=1.0,
        skin_strength=-1.0,
        nr_style="Default",
        auto_mask=False,
        **kwargs,
    ):
        if not enable:
            return

        pp.image = dlss5_manager.enhance_pil_image(
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

        pp.info["DLSS 5"] = f"Scale={scale_mode}, Preset={model_preset}, NRI={nr_intensity}"
