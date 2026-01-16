#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
DEPRECATED: Blender Installer for ComfyUI-GeometryPack

This script is DEPRECATED. Blender functionality now uses bpy (Blender as Python
module) via comfy-env isolation, which is much cleaner and doesn't require
downloading ~800MB of full Blender.

Instead of running this script, use comfy-env:

    cd custom_nodes/ComfyUI-GeometryPack
    comfy-env install

This creates an isolated Python 3.11 environment with:
  - bpy>=4.2,<4.3 (Blender as Python module)
  - cumesh (GPU-accelerated mesh operations)
  - numpy<2 (required by bpy)
  - trimesh>=4.0.0

The old subprocess-based Blender approach is kept for backwards compatibility
but will be removed in a future version.

Usage (DEPRECATED):
    python blender_install.py
"""

import warnings
warnings.warn(
    "blender_install.py is DEPRECATED. "
    "Use 'comfy-env install' instead for bpy + cumesh in an isolated environment. "
    "See comfy-env.toml for configuration.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
import platform
import urllib.request
import tarfile
import zipfile
import shutil
import subprocess
from pathlib import Path

# Try to import optimized libraries, fallback to basic if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def get_blender_download_url(platform_name, architecture):
    """
    Get Blender 4.2 LTS download URL for the platform.

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    urls = {
        ("linux", "x64"): (
            f"{base_url}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{base_url}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{base_url}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{base_url}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        print(f"[Blender] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file_optimized(url, dest_path):
    """Download file with requests and tqdm for better performance and progress."""
    print(f"[Blender] Downloading: {url}")
    print(f"[Blender] Destination: {dest_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        if HAS_TQDM:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        else:
            downloaded = 0
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            sys.stdout.write(f"\r[Blender] Progress: {percent}%")
                            sys.stdout.flush()
            sys.stdout.write("\n")

        print("[Blender] Download complete!")
        return True
    except Exception as e:
        print(f"\n[Blender] Error downloading: {e}")
        return False


def download_file(url, dest_path):
    """Download file with progress."""
    if HAS_REQUESTS:
        return download_file_optimized(url, dest_path)

    print(f"[Blender] Downloading: {url}")
    print(f"[Blender] Destination: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[Blender] Progress: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        print("[Blender] Download complete!")
        return True
    except Exception as e:
        print(f"\n[Blender] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS) - sequential only."""
    print(f"[Blender] Extracting: {archive_path}")

    try:
        if archive_path.endswith('.tar.xz'):
            print("[Blender] Extracting .tar.xz archive...")
            with tarfile.open(archive_path, 'r:*') as tar:
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(extract_to, filter='data')
                else:
                    tar.extractall(extract_to)

        elif archive_path.endswith(('.tar.gz', '.tar.bz2')):
            print("[Blender] Extracting tar archive...")
            with tarfile.open(archive_path, 'r:*') as tar:
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(extract_to, filter='data')
                else:
                    tar.extractall(extract_to)

        elif archive_path.endswith('.zip'):
            print("[Blender] Extracting .zip archive (this may take a moment)...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        elif archive_path.endswith('.dmg'):
            print("[Blender] DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[Blender] Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[Blender] Error: Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[Blender] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[Blender] Error: Blender.app not found in {mount_point}")
                    return False
            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[Blender] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[Blender] Extraction complete!")
        return True

    except Exception as e:
        print(f"[Blender] Error extracting: {e}")
        return False


def main():
    """Main installation function."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Blender Installation")
    print("="*60 + "\n")

    script_dir = Path(__file__).parent.absolute()
    blender_dir = script_dir / "_blender"

    if blender_dir.exists():
        print("[Blender] Blender directory already exists at:")
        print(f"[Blender]   {blender_dir}")
        print("[Blender] Skipping download. Delete '_blender/' folder to reinstall.")
        return True

    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[Blender] Error: Could not detect platform")
        print("[Blender] Please install Blender manually from: https://www.blender.org/download/")
        return False

    print(f"[Blender] Detected platform: {plat}-{arch}")

    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[Blender] Error: Could not find Blender download for your platform")
        print("[Blender] Please install Blender manually from: https://www.blender.org/download/")
        return False

    temp_dir = script_dir / "_temp_download"
    temp_dir.mkdir(exist_ok=True)

    try:
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return False

        blender_dir.mkdir(exist_ok=True)
        if not extract_archive(str(download_path), str(blender_dir)):
            return False

        print("\n[Blender] Installation complete!")
        print(f"[Blender] Location: {blender_dir}")

        if plat == "windows":
            blender_exe = list(blender_dir.rglob("blender.exe"))
        else:
            blender_exe = list(blender_dir.rglob("blender"))

        if blender_exe:
            print(f"[Blender] Blender executable: {blender_exe[0]}")

        return True

    except Exception as e:
        print(f"\n[Blender] Error during installation: {e}")
        return False

    finally:
        if temp_dir.exists():
            print("[Blender] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
