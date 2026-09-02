import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

import launch

# Ensure dependencies
if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python-headless", "opencv-python for DLSS 5 optical flow")

if not launch.is_installed("av"):
    launch.run_pip("install av", "PyAV for DLSS 5 video demuxing")

# Check runtime binaries
runtime_dir = Path(__file__).resolve().parent / "bin" / "runtime"
runtime_dir.mkdir(parents=True, exist_ok=True)

required_dlls = [
    "nvngx.dll",
    "dxgi.dll",
    "renodx-dlss5.addon64",
    "nvngx_dlssnr.dll",
    "nvngx_dlss.dll",
]

missing = [f for f in required_dlls if not (runtime_dir / f).exists()]
if missing:
    print(f"[sd-forge-dlss5] Missing runtime binaries: {missing}")
    print("[sd-forge-dlss5] Downloading official DLSS 5 D3D12 runtime worker package from GitHub...")
    release_url = "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/v3.0/dlss5-visual-enhancer-v3.0.zip"
    zip_path = runtime_dir / "dlss5-runtime.zip"
    try:
        urllib.request.urlretrieve(release_url, str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if filename in required_dlls:
                    with zf.open(member) as src, open(runtime_dir / filename, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        if zip_path.exists():
            zip_path.unlink()
        print("[sd-forge-dlss5] DLSS 5 D3D12 runtime binaries successfully installed.")
    except Exception as e:
        print(f"[sd-forge-dlss5] Warning: Failed to auto-download runtime binaries: {e}")
        print("[sd-forge-dlss5] Please download and place nvngx.dll, dxgi.dll, renodx-dlss5.addon64, nvngx_dlssnr.dll into bin/runtime/")
