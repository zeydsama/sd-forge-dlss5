import struct
from typing import NamedTuple, Optional

# Magic Byte Constants
VIDEO_MAGIC = 0x34563544       # 'D5V4' in ASCII (Little-Endian)
SETUP_RESPONSE_MAGIC = 0x34505553  # 'SUP4' in ASCII
FRAME_MAGIC = 0x314D5246       # 'FRM1' in ASCII
OUT_MAGIC = 0x3154554F         # 'OUT1' in ASCII

# Struct Formats
# 14 uint32 + 4 float32 = 72 bytes
VIDEO_HEADER_FORMAT = "<14I4f"
# 12 uint32 = 48 bytes
SETUP_RESPONSE_FORMAT = "<12I"
# 4 uint32 + 1 int64 = 24 bytes
FRAME_HEADER_FORMAT = "<4Iq"
# 5 uint32 + 1 int64 = 28 bytes
OUT_HEADER_FORMAT = "<5Iq"

# Perf Quality / Scale Presets
PERF_QUALITY_MAP = {
    "DLAA (1.0x)": 5,
    "DLAA (Native)": 5,
    "Quality (1.5x)": 2,
    "Balanced (1.72x)": 1,
    "Performance (2.0x)": 0,
    "Ultra Performance (3.0x)": 3,
}

SCALE_FACTOR_MAP = {
    "DLAA (1.0x)": 1.0,
    "DLAA (Native)": 1.0,
    "Quality (1.5x)": 1.5,
    "Balanced (1.72x)": 1.724,
    "Performance (2.0x)": 2.0,
    "Ultra Performance (3.0x)": 3.0,
}

# DLSS Model Presets
DLSS_MODEL_PRESET_MAP = {
    "Default": 0,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
}

# Neural Rendering Styles
NR_STYLE_MAP = {
    "Default": 0,
    "Natural": 1,
    "Cinematic": 2,
}

# NR Presets
NR_PRESET_MAP = {
    "Default": 0,
    "Preset #1": 1,
    "Preset #2": 2,
    "Preset #3": 3,
}


class SetupResponse(NamedTuple):
    setup_magic: int
    setup_ok: int
    setup_result: int
    render_width: int
    render_height: int
    output_width: int
    output_height: int
    min_w: int
    min_h: int
    max_w: int
    max_h: int
    applied_dlss_model_preset: int


class OutHeader(NamedTuple):
    out_magic: int
    out_index: int
    ok: int
    byte_count: int
    ngx_result: int
    out_pts: int


def pack_video_header(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    warmup_frames: int = 0,
    frame_count: int = 1,
    perf_quality: int = 2,
    dlss_model_preset: int = 0,
    profile: int = 0,
    preset: int = 0,
    style: int = 0,
    auto_mask: int = 0,
    ui_correction: int = 0,
    intensity: float = 1.0,
    local_tone: float = 1.0,
    local_structure: float = 1.0,
    skin_structure: float = -1.0,
) -> bytes:
    """Pack the initial 72-byte D5V4 setup header sent to the D3D12 worker."""
    return struct.pack(
        VIDEO_HEADER_FORMAT,
        VIDEO_MAGIC,
        int(input_width),
        int(input_height),
        int(output_width),
        int(output_height),
        int(warmup_frames),
        int(frame_count),
        int(perf_quality),
        int(dlss_model_preset),
        int(profile),
        int(preset),
        int(style),
        int(auto_mask),
        int(ui_correction),
        float(intensity),
        float(local_tone),
        float(local_structure),
        float(skin_structure),
    )


def unpack_setup_response(data: bytes) -> SetupResponse:
    """Unpack the 48-byte setup response from the worker."""
    if len(data) != struct.calcsize(SETUP_RESPONSE_FORMAT):
        raise ValueError(f"Invalid setup response size: expected {struct.calcsize(SETUP_RESPONSE_FORMAT)}, got {len(data)}")
    unpacked = struct.unpack(SETUP_RESPONSE_FORMAT, data)
    return SetupResponse(*unpacked)


def pack_frame_header(
    frame_index: int,
    reset: int = 0,
    flags: int = 0,
    pts: int = 0,
) -> bytes:
    """Pack the 24-byte per-frame FRM1 header."""
    return struct.pack(
        FRAME_HEADER_FORMAT,
        FRAME_MAGIC,
        int(frame_index),
        int(reset),
        int(flags),
        int(pts),
    )


def unpack_out_header(data: bytes) -> OutHeader:
    """Unpack the 28-byte per-frame OUT1 response header from the worker."""
    if len(data) != struct.calcsize(OUT_HEADER_FORMAT):
        raise ValueError(f"Invalid out header size: expected {struct.calcsize(OUT_HEADER_FORMAT)}, got {len(data)}")
    unpacked = struct.unpack(OUT_HEADER_FORMAT, data)
    return OutHeader(*unpacked)
