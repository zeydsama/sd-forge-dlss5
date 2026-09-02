import logging
import torch

logger = logging.getLogger("dlss5.vram")


def prepare_vram_for_dlss(min_free_vram_gb: float = 1.0):
    """
    Coordinates PyTorch VRAM with Forge before launching or streaming into the D3D12 DLSS worker.
    Frees cached allocations so DirectX 12 Tensor Core heaps have clean headroom.
    """
    try:
        from modules import devices
        devices.torch_gc()
    except Exception as e:
        logger.debug(f"devices.torch_gc call skipped: {e}")

    try:
        from backend import memory_management
        memory_management.soft_empty_cache()
    except Exception as e:
        logger.debug(f"soft_empty_cache skipped: {e}")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def restore_vram_state():
    """Called after DLSS 5 passes to ensure PyTorch allocator is ready for further diffusion steps."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
