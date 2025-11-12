#!/usr/bin/env python3
"""
GeometryPack Installer
Automatically downloads and installs Blender for UV unwrapping and remeshing nodes.
"""

import os
import sys
import platform
import urllib.request
import json
import tarfile
import zipfile
import shutil
from pathlib import Path


def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform names
    if system == "darwin":
        plat = "macos"
        if machine == "arm64":
            arch = "arm64"
        else:
            arch = "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"  # Most common
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

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    # Platform-specific URLs for Blender 4.2.3 LTS
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
        print(f"[Install] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"[Install] Downloading: {url}")
    print(f"[Install] Destination: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[Install] Progress: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        print("[Install] Download complete!")
        return True
    except Exception as e:
        print(f"\n[Install] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    print(f"[Install] Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.dmg'):
            # macOS DMG - mount and copy Blender.app
            print("[Install] DMG detected - mounting disk image...")
            import subprocess

            # Mount the DMG
            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[Install] Error mounting DMG: {mount_result.stderr}")
                return False

            # Find the mount point from the output
            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[Install] Error: Could not find mount point")
                return False

            try:
                # Copy Blender.app to destination
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[Install] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[Install] Error: Blender.app not found in {mount_point}")
                    return False

            finally:
                # Unmount the DMG
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[Install] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[Install] Extraction complete!")
        return True

    except Exception as e:
        print(f"[Install] Error extracting: {e}")
        return False


def install_blender():
    """Main installation function."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Blender Installation")
    print("="*60 + "\n")

    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    blender_dir = script_dir / "_blender"

    # Check if Blender already installed
    if blender_dir.exists():
        print("[Install] Blender directory already exists at:")
        print(f"[Install]   {blender_dir}")
        print("[Install] Skipping download. Delete '_blender/' folder to reinstall.")
        return True

    # Detect platform
    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[Install] Error: Could not detect platform")
        print("[Install] Please install Blender manually from: https://www.blender.org/download/")
        return False

    print(f"[Install] Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[Install] Error: Could not find Blender download for your platform")
        print("[Install] Please install Blender manually from: https://www.blender.org/download/")
        return False

    # Create temporary download directory
    temp_dir = script_dir / "_temp_download"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return False

        # Extract
        blender_dir.mkdir(exist_ok=True)
        if not extract_archive(str(download_path), str(blender_dir)):
            return False

        print("\n[Install] ✓ Blender installation complete!")
        print(f"[Install] Location: {blender_dir}")

        # Find blender executable
        if plat == "windows":
            blender_exe = list(blender_dir.rglob("blender.exe"))
        else:
            blender_exe = list(blender_dir.rglob("blender"))

        if blender_exe:
            print(f"[Install] Blender executable: {blender_exe[0]}")

        return True

    except Exception as e:
        print(f"\n[Install] Error during installation: {e}")
        return False

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            print("[Install] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Entry point."""
    success = install_blender()

    if success:
        print("\n" + "="*60)
        print("Installation completed successfully!")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Installation failed.")
        print("You can:")
        print("  1. Install Blender manually: https://www.blender.org/download/")
        print("  2. Try running this script again: python install.py")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
