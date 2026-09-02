import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger("dlss5.runtime")

# Required DLLs in the runtime directory
REQUIRED_BINARIES = [
    "nvngx.dll",
    "dxgi.dll",
    "renodx-dlss5.addon64",
    "nvngx_dlssnr.dll",
    "nvngx_dlss.dll",
]


def get_default_runtime_dir() -> Path:
    """Returns the absolute path to the bundled bin/runtime folder."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "bin" / "runtime"


def find_runtime_dir(custom_path: str | None = None) -> Path:
    """Resolves the active runtime directory from custom argument or default path."""
    if custom_path and os.path.isdir(custom_path):
        return Path(custom_path)
    return get_default_runtime_dir()


def validate_runtime_binaries(runtime_dir: Path | None = None) -> tuple[bool, list[str]]:
    """
    Checks if all required DLLs and add-ons are present in the runtime directory.
    Returns (is_valid, list_of_missing_files).
    """
    if runtime_dir is None:
        runtime_dir = get_default_runtime_dir()

    if not runtime_dir.exists():
        return False, [f"Directory not found: {runtime_dir}"]

    missing = []
    for binary in REQUIRED_BINARIES:
        binary_path = runtime_dir / binary
        if not binary_path.exists() or binary_path.stat().st_size == 0:
            missing.append(binary)

    return (len(missing) == 0), missing


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a binary file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest().upper()
