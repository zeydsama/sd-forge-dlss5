import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .hardware_validator import check_hardware_compatibility
from .optical_flow import OpticalFlowGenerator
from .protocol_v4 import (
    DLSS_MODEL_PRESET_MAP,
    NR_PRESET_MAP,
    NR_STYLE_MAP,
    OUT_MAGIC,
    PERF_QUALITY_MAP,
    SCALE_FACTOR_MAP,
    SETUP_RESPONSE_MAGIC,
    OutHeader,
    SetupResponse,
    pack_frame_header,
    pack_video_header,
    unpack_out_header,
    unpack_setup_response,
)
from .runtime_manager import find_runtime_dir, validate_runtime_binaries
from .vram_coordinator import prepare_vram_for_dlss, restore_vram_state

logger = logging.getLogger("dlss5.worker")


class DLSS5WorkerSession:
    """Manages an active D3D12 worker subprocess and handles binary Protocol v4 streaming."""

    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir
        self.process: Optional[subprocess.Popen] = None
        self.current_w: int = 0
        self.current_h: int = 0
        self.output_w: int = 0
        self.output_h: int = 0
        self.frame_index: int = 0
        self.flow_gen = OpticalFlowGenerator()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start_worker(
        self,
        input_w: int,
        input_h: int,
        output_w: int,
        output_h: int,
        scale_mode: str = "Quality (1.5x)",
        model_preset: str = "Default",
        nr_intensity: float = 1.0,
        tone_strength: float = 1.0,
        struct_strength: float = 1.0,
        skin_strength: float = -1.0,
        nr_style: str = "Default",
        nr_preset: str = "Default",
        auto_mask: bool = False,
    ) -> bool:
        """Starts the D3D12 worker process and conducts the initial setup handshake."""
        self.stop_worker()

        # Locate worker binary (supports standalone worker or nvngx.dll host)
        worker_exe = self.runtime_dir / "nvngx.dll"
        if not worker_exe.exists():
            worker_exe = self.runtime_dir / "nvngx_worker.exe"

        if not worker_exe.exists():
            logger.info(f"DLSS 5 native worker host not present at {self.runtime_dir}. Falling back to standard resampling.")
            return False

        perf_quality = PERF_QUALITY_MAP.get(scale_mode, 2)
        dlss_model = DLSS_MODEL_PRESET_MAP.get(model_preset, 0)
        nr_style_val = NR_STYLE_MAP.get(nr_style, 0)
        nr_preset_val = NR_PRESET_MAP.get(nr_preset, 0)
        auto_mask_val = 1 if auto_mask else 0

        # Pack initial 72-byte setup packet
        setup_pkt = pack_video_header(
            input_width=input_w,
            input_height=input_h,
            output_width=output_w,
            output_height=output_h,
            warmup_frames=0,
            frame_count=1,
            perf_quality=perf_quality,
            dlss_model_preset=dlss_model,
            profile=0,
            preset=nr_preset_val,
            style=nr_style_val,
            auto_mask=auto_mask_val,
            ui_correction=0,
            intensity=nr_intensity,
            local_tone=tone_strength,
            local_structure=struct_strength,
            skin_structure=skin_strength,
        )

        try:
            # Launch worker in runtime directory
            self.process = subprocess.Popen(
                [str(worker_exe), "--video"],
                cwd=str(self.runtime_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=0x08000000 if os.name == "nt" else 0,  # CREATE_NO_WINDOW
            )

            # Send setup header
            self.process.stdin.write(setup_pkt)
            self.process.stdin.flush()

            # Read 48-byte setup response
            resp_bytes = self.process.stdout.read(48)
            if len(resp_bytes) < 48:
                err = self.process.stderr.read().decode("utf-8", errors="ignore")
                logger.error(f"DLSS 5 setup handshake failed. Stderr: {err}")
                self.stop_worker()
                return False

            resp = unpack_setup_response(resp_bytes)
            if resp.setup_magic != SETUP_RESPONSE_MAGIC or resp.setup_ok != 1:
                logger.error(f"Worker setup rejected: magic={hex(resp.setup_magic)}, ok={resp.setup_ok}, result={resp.setup_result}")
                self.stop_worker()
                return False

            self.current_w = input_w
            self.current_h = input_h
            self.output_w = output_w
            self.output_h = output_h
            self.frame_index = 0
            self.flow_gen.reset()

            logger.info(f"DLSS 5 Worker Session Initialized: {input_w}x{input_h} -> {output_w}x{output_h}")
            return True

        except Exception as e:
            logger.error(f"Failed to start DLSS 5 worker: {e}", exc_info=True)
            self.stop_worker()
            return False

    def enhance_frame(
        self,
        frame_rgba: np.ndarray,
        reset_history: bool = True,
        motion_vectors: Optional[bytes] = None,
    ) -> Optional[np.ndarray]:
        """
        Sends a single RGBA8 frame to the active worker and returns the enhanced RGBA8 frame.
        """
        if not self.is_alive():
            logger.error("DLSS 5 worker is not running.")
            return None

        h, w = frame_rgba.shape[:2]
        if w != self.current_w or h != self.current_h:
            logger.error(f"Frame dimension mismatch: expected {self.current_w}x{self.current_h}, got {w}x{h}")
            return None

        # Prepare motion vectors (zero for still images, or computed for videos)
        if motion_vectors is None:
            if reset_history:
                mv_bytes = OpticalFlowGenerator.create_zero_motion_vectors(w, h)
            else:
                mv_bytes = self.flow_gen.compute_motion_vectors(frame_rgba)
        else:
            mv_bytes = motion_vectors

        # Pack 24-byte frame header
        frame_hdr = pack_frame_header(
            frame_index=self.frame_index,
            reset=1 if reset_history else 0,
            flags=0,
            pts=int(time.time() * 1000),
        )

        try:
            # 1. Write Header + RGBA8 Pixel Buffer + Motion Vectors
            raw_bytes = frame_rgba.tobytes()
            self.process.stdin.write(frame_hdr)
            self.process.stdin.write(raw_bytes)
            self.process.stdin.write(mv_bytes)
            self.process.stdin.flush()

            # 2. Read 28-byte OUT header
            out_hdr_bytes = self.process.stdout.read(28)
            if len(out_hdr_bytes) < 28:
                logger.error("DLSS 5 worker EOF or crash while reading output header.")
                return None

            out_hdr = unpack_out_header(out_hdr_bytes)
            if out_hdr.out_magic != OUT_MAGIC or out_hdr.ok != 1:
                logger.error(f"DLSS 5 worker returned error: ok={out_hdr.ok}, ngx_result={out_hdr.ngx_result}")
                return None

            # 3. Read output image bytes
            expected_bytes = self.output_w * self.output_h * 4
            out_raw = self.process.stdout.read(expected_bytes)
            if len(out_raw) < expected_bytes:
                logger.error(f"Truncated frame data: expected {expected_bytes}, got {len(out_raw)}")
                return None

            self.frame_index += 1

            # Convert back to uint8 RGBA numpy array
            out_array = np.frombuffer(out_raw, dtype=np.uint8).reshape((self.output_h, self.output_w, 4))
            return out_array

        except Exception as e:
            logger.error(f"Error during frame streaming: {e}", exc_info=True)
            return None

    def stop_worker(self):
        """Terminates the worker process safely."""
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None


class DLSS5Manager:
    """High-level facade for executing DLSS 5 enhancements on PIL Images and frame sequences."""

    def __init__(self, custom_runtime_dir: Optional[str] = None):
        self.runtime_dir = find_runtime_dir(custom_runtime_dir)
        self.session: Optional[DLSS5WorkerSession] = None
        self._last_config: dict = {}

    def calculate_dimensions(self, width: int, height: int, scale_mode: str) -> tuple[int, int]:
        """Calculates even-integer output resolution according to scale mode."""
        scale = SCALE_FACTOR_MAP.get(scale_mode, 1.0)
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))

        # DLSS requires even dimensions
        out_w = out_w if out_w % 2 == 0 else out_w + 1
        out_h = out_h if out_h % 2 == 0 else out_h + 1
        return out_w, out_h

    def _get_or_create_session(
        self,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        scale_mode: str,
        model_preset: str,
        nr_intensity: float,
        tone_strength: float,
        struct_strength: float,
        skin_strength: float,
        nr_style: str,
        auto_mask: bool,
    ) -> Optional[DLSS5WorkerSession]:
        config = {
            "in_w": in_w,
            "in_h": in_h,
            "out_w": out_w,
            "out_h": out_h,
            "scale_mode": scale_mode,
            "model_preset": model_preset,
            "nr_intensity": nr_intensity,
            "tone_strength": tone_strength,
            "struct_strength": struct_strength,
            "skin_strength": skin_strength,
            "nr_style": nr_style,
            "auto_mask": auto_mask,
        }

        # Reuse existing warm session if process is alive and config is identical
        if self.session is not None and self.session.is_alive() and self._last_config == config:
            return self.session

        # Start new or reconfigured session
        session = DLSS5WorkerSession(self.runtime_dir)
        started = session.start_worker(
            input_w=in_w,
            input_h=in_h,
            output_w=out_w,
            output_h=out_h,
            scale_mode=scale_mode,
            model_preset=model_preset,
            nr_intensity=nr_intensity,
            tone_strength=tone_strength,
            struct_strength=struct_strength,
            skin_strength=skin_strength,
            nr_style=nr_style,
            auto_mask=auto_mask,
        )

        if not started:
            return None

        # Clean up old session
        if self.session is not None and self.session != session:
            self.session.stop_worker()

        self.session = session
        self._last_config = config
        return self.session

    def enhance_pil_image(
        self,
        image: Image.Image,
        scale_mode: str = "Quality (1.5x)",
        model_preset: str = "Default",
        nr_intensity: float = 1.0,
        tone_strength: float = 1.0,
        struct_strength: float = 1.0,
        skin_strength: float = -1.0,
        nr_style: str = "Default",
        auto_mask: bool = False,
    ) -> Image.Image:
        """
        Enhances a PIL Image using DLSS 5 Neural Rendering.
        If D3D12 binaries are missing or hardware is non-RTX, falls back gracefully to Lanczos resampling.
        """
        # 1. Check hardware compatibility
        hw_ok, hw_msg = check_hardware_compatibility()
        if not hw_ok:
            logger.warning(f"DLSS 5 bypassed: {hw_msg}")
            return self._fallback_upscale(image, scale_mode)

        # 2. Check runtime binaries
        bin_ok, missing = validate_runtime_binaries(self.runtime_dir)
        if not bin_ok:
            logger.warning(f"DLSS 5 binaries missing: {missing}. Falling back to standard resampling.")
            return self._fallback_upscale(image, scale_mode)

        # 3. Coordinate VRAM
        prepare_vram_for_dlss()

        # 4. Convert input to RGBA numpy
        orig_mode = image.mode
        img_rgba = image.convert("RGBA")
        in_w, in_h = img_rgba.size

        # Round input dimensions to even integers if needed
        in_w_even = in_w if in_w % 2 == 0 else in_w - 1
        in_h_even = in_h if in_h % 2 == 0 else in_h - 1
        if (in_w_even, in_h_even) != (in_w, in_h):
            img_rgba = img_rgba.resize((in_w_even, in_h_even), Image.Resampling.LANCZOS)
            in_w, in_h = in_w_even, in_h_even

        out_w, out_h = self.calculate_dimensions(in_w, in_h, scale_mode)
        arr_in = np.array(img_rgba)

        # 5. Get or start warm worker session
        session = self._get_or_create_session(
            in_w=in_w,
            in_h=in_h,
            out_w=out_w,
            out_h=out_h,
            scale_mode=scale_mode,
            model_preset=model_preset,
            nr_intensity=nr_intensity,
            tone_strength=tone_strength,
            struct_strength=struct_strength,
            skin_strength=skin_strength,
            nr_style=nr_style,
            auto_mask=auto_mask,
        )

        if session is None:
            logger.warning("DLSS 5 worker failed to start. Falling back.")
            return self._fallback_upscale(image, scale_mode)

        try:
            enhanced_arr = session.enhance_frame(arr_in, reset_history=True)
            if enhanced_arr is None:
                logger.warning("Worker returned empty frame. Falling back.")
                return self._fallback_upscale(image, scale_mode)

            result_img = Image.fromarray(enhanced_arr, mode="RGBA")
            if orig_mode == "RGB":
                result_img = result_img.convert("RGB")
            return result_img

        finally:
            restore_vram_state()

    def enhance_frame_sequence(
        self,
        frames: list[np.ndarray],
        scale_mode: str = "Quality (1.5x)",
        model_preset: str = "Default",
        nr_intensity: float = 1.0,
        tone_strength: float = 1.0,
        struct_strength: float = 1.0,
        skin_strength: float = -1.0,
        nr_style: str = "Default",
        auto_mask: bool = False,
    ) -> list[np.ndarray]:
        """Enhances a sequence of video frames with temporal optical flow."""
        if not frames:
            return frames

        hw_ok, _ = check_hardware_compatibility()
        bin_ok, _ = validate_runtime_binaries(self.runtime_dir)
        if not hw_ok or not bin_ok:
            return frames

        prepare_vram_for_dlss()

        first_frame = frames[0]
        h, w = first_frame.shape[:2]
        out_w, out_h = self.calculate_dimensions(w, h, scale_mode)

        session = DLSS5WorkerSession(self.runtime_dir)
        started = session.start_worker(
            input_w=w,
            input_h=h,
            output_w=out_w,
            output_h=out_h,
            scale_mode=scale_mode,
            model_preset=model_preset,
            nr_intensity=nr_intensity,
            tone_strength=tone_strength,
            struct_strength=struct_strength,
            skin_strength=skin_strength,
            nr_style=nr_style,
            auto_mask=auto_mask,
        )

        if not started:
            return frames

        output_frames = []
        try:
            for i, frame in enumerate(frames):
                # Ensure RGBA
                if frame.shape[2] == 3:
                    frame_rgba = np.dstack([frame, np.full((h, w), 255, dtype=np.uint8)])
                else:
                    frame_rgba = frame

                # Reset history on first frame, continue temporal flow on subsequent frames
                enhanced = session.enhance_frame(frame_rgba, reset_history=(i == 0))
                if enhanced is not None:
                    if frame.shape[2] == 3:
                        output_frames.append(enhanced[:, :, :3])
                    else:
                        output_frames.append(enhanced)
                else:
                    output_frames.append(frame)

            return output_frames

        finally:
            session.stop_worker()
            restore_vram_state()

    def _fallback_upscale(self, image: Image.Image, scale_mode: str) -> Image.Image:
        scale = SCALE_FACTOR_MAP.get(scale_mode, 1.0)
        if math.isclose(scale, 1.0):
            return image
        out_w = int(round(image.width * scale))
        out_h = int(round(image.height * scale))
        return image.resize((out_w, out_h), Image.Resampling.LANCZOS)


# Global Singleton Manager
dlss5_manager = DLSS5Manager()
