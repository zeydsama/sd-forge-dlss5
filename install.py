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

# Permanent GitHub Release download link (Streamline 2.13 / DLSS 3.10 runtime package)
PRIMARY_RUNTIME_URL = "https://github.com/zeydsama/sd-forge-dlss5/releases/download/v1.0.0/DLSS310.8.0-Streamline2.13.zip"
FALLBACK_RUNTIME_URL = "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/3.0/DLSS.5.Visual.Enhancer.v3.0.zip"

missing = [f for f in required_dlls if not (runtime_dir / f).exists()]
if missing:
    print(f"[sd-forge-dlss5] Missing runtime binaries: {missing}")
    print("[sd-forge-dlss5] Auto-downloading official DLSS / Streamline runtime binaries...")

    downloaded = False
    for url_label, url in [("primary", PRIMARY_RUNTIME_URL), ("fallback", FALLBACK_RUNTIME_URL)]:
        zip_path = runtime_dir / "dlss5-runtime.zip"
        try:
            print(f"[sd-forge-dlss5] Fetching package from {url_label} source...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)

            print("[sd-forge-dlss5] Extracting runtime binaries...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    # Extract any DLL or required binary into runtime directory
                    if filename.endswith(".dll") or filename in required_dlls:
                        with zf.open(member) as src, open(runtime_dir / filename, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        print(f"[sd-forge-dlss5]   + Extracted: {filename}")

            if zip_path.exists():
                zip_path.unlink()

            # Verify if missing files are now satisfied
            remaining = [f for f in required_dlls if not (runtime_dir / f).exists()]
            if not remaining:
                print("[sd-forge-dlss5] All DLSS 5 runtime binaries successfully installed.")
                downloaded = True
                break
            else:
                print(f"[sd-forge-dlss5] Package extracted, still awaiting: {remaining}")
        except Exception as e:
            print(f"[sd-forge-dlss5] Download failed from {url_label} source: {e}")
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass

    if not downloaded:
        still_missing = [f for f in required_dlls if not (runtime_dir / f).exists()]
        if still_missing:
            print(f"[sd-forge-dlss5] Warning: Runtime binaries incomplete ({still_missing}).")
            print("[sd-forge-dlss5] Please place nvngx.dll, dxgi.dll, renodx-dlss5.addon64, nvngx_dlssnr.dll into bin/runtime/")
