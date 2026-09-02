import logging
import os
import platform
import subprocess
import torch

logger = logging.getLogger("dlss5.hardware")


def check_hardware_compatibility() -> tuple[bool, str]:
    """
    Checks if current environment meets the requirements for DLSS 5 Neural Rendering:
    1. 64-bit Windows OS (DirectX 12 Agility requirement).
    2. NVIDIA GeForce RTX series GPU (Compute Capability >= 7.5: Turing, Ampere, Ada Lovelace, Blackwell).
    """
    # 1. OS check
    if platform.system() != "Windows":
        return False, f"DLSS 5 requires Windows 10/11 64-bit with DirectX 12 support (current OS: {platform.system()})."

    # 2. CUDA & GPU check
    if not torch.cuda.is_available():
        return False, "CUDA is not available. DLSS 5 requires an NVIDIA RTX GPU."

    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, "No CUDA devices detected."

        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        major, minor = capability

        # DLSS Neural Rendering requires RTX architecture (SM 7.5+)
        if major < 7 or (major == 7 and minor < 5):
            return False, f"GPU '{device_name}' (Compute Capability {major}.{minor}) is not supported. DLSS 5 requires RTX 20/30/40/50 series (Compute Capability >= 7.5)."

        logger.info(f"DLSS 5 Hardware Check Passed: {device_name} (Compute Capability {major}.{minor})")
        return True, f"Compatible ({device_name})"

    except Exception as e:
        logger.warning(f"Hardware validation encountered an exception: {e}")
        return False, f"Hardware check error: {e}"
