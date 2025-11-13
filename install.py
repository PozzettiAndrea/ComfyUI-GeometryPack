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
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def check_tool_available(tool_name):
    """Check if a system tool is available in PATH."""
    try:
        result = subprocess.run(
            ['which', tool_name] if os.name != 'nt' else ['where', tool_name],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def get_cpu_count():
    """Get number of CPU cores for parallel processing."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


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


def download_file_optimized(url, dest_path):
    """Download file with requests and tqdm for better performance and progress."""
    print(f"[Install] Downloading: {url}")
    print(f"[Install] Destination: {dest_path}")

    try:
        # Stream download with requests
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB chunks

        if HAS_TQDM:
            # Use tqdm progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        else:
            # Fallback to basic progress
            downloaded = 0
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            sys.stdout.write(f"\r[Install] Progress: {percent}%")
                            sys.stdout.flush()
            sys.stdout.write("\n")

        print("[Install] Download complete!")
        return True
    except Exception as e:
        print(f"\n[Install] Error downloading: {e}")
        return False


def download_file(url, dest_path):
    """Download file with progress (uses optimized version if available)."""
    if HAS_REQUESTS:
        return download_file_optimized(url, dest_path)

    # Fallback to urllib
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


def extract_tar_xz_optimized(archive_path, extract_to):
    """Extract .tar.xz using pixz for multi-threaded decompression."""
    cpu_cores = get_cpu_count()

    # Check if pixz is available
    has_pixz = check_tool_available('pixz')

    if has_pixz:
        print(f"[Install] Using pixz with {cpu_cores} cores for fast extraction...")
        try:
            # Use pixz to decompress, then tar to extract
            # pixz -d -p <cores> archive.tar.xz extracts to archive.tar
            tar_path = archive_path.replace('.tar.xz', '.tar')

            # Decompress with pixz
            result = subprocess.run(
                ['pixz', '-d', '-p', str(cpu_cores), archive_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"[Install] pixz failed: {result.stderr}")
                return False

            # Extract the tar file
            with tarfile.open(tar_path, 'r:') as tar:
                # Use filter for Python 3.14+ compatibility
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(extract_to, filter='data')
                else:
                    tar.extractall(extract_to)

            # Clean up the intermediate .tar file
            if os.path.exists(tar_path):
                os.remove(tar_path)

            return True
        except Exception as e:
            print(f"[Install] Error with pixz extraction: {e}")
            return False
    else:
        # Fallback to standard tarfile
        print("[Install] pixz not found, using standard extraction (slower)...")
        try:
            with tarfile.open(archive_path, 'r:*') as tar:
                # Use filter for Python 3.14+ compatibility
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(extract_to, filter='data')
                else:
                    tar.extractall(extract_to)
            return True
        except Exception as e:
            print(f"[Install] Error with standard extraction: {e}")
            return False


def extract_zip_parallel(archive_path, extract_to):
    """Extract .zip using parallel extraction for better performance."""
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            # For small archives, parallel extraction overhead isn't worth it
            if len(members) < 10:
                zip_ref.extractall(extract_to)
                return True

            print(f"[Install] Extracting {len(members)} files in parallel...")

            # Extract files in parallel
            def extract_member(member):
                try:
                    zip_ref.extract(member, extract_to)
                    return True
                except Exception as e:
                    print(f"[Install] Error extracting {member}: {e}")
                    return False

            cpu_cores = get_cpu_count()
            with ThreadPoolExecutor(max_workers=min(cpu_cores, 8)) as executor:
                results = list(executor.map(extract_member, members))

            # Check if all extractions succeeded
            if not all(results):
                print("[Install] Some files failed to extract")
                return False

            return True
    except Exception as e:
        print(f"[Install] Error with parallel zip extraction: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS) with optimizations."""
    print(f"[Install] Extracting: {archive_path}")

    try:
        if archive_path.endswith('.tar.xz'):
            # Use optimized extraction for .tar.xz
            if not extract_tar_xz_optimized(archive_path, extract_to):
                return False
        elif archive_path.endswith(('.tar.gz', '.tar.bz2')):
            # Standard extraction for other tar formats
            with tarfile.open(archive_path, 'r:*') as tar:
                # Use filter for Python 3.14+ compatibility
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(extract_to, filter='data')
                else:
                    tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            # Use parallel extraction for .zip
            if not extract_zip_parallel(archive_path, extract_to):
                return False
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


def install_system_dependencies():
    """Install required system dependencies (Linux only)."""
    plat, _ = get_platform_info()

    if plat != "linux":
        return True

    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: System Dependencies")
    print("="*60 + "\n")

    print("[Install] Checking for required OpenGL libraries...")
    print("[Install] These are needed for PyMeshLab remeshing to work properly.")

    # Check if running as root or with sudo
    is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False

    try:
        # Try to install OpenGL libraries
        # Note: Package names changed in Ubuntu 24.04+
        # Old: libgl1-mesa-glx (pre-24.04)
        # New: libgl1, libglx-mesa0 (24.04+)
        packages = ["libgl1", "libglu1-mesa", "libglx-mesa0", "libosmesa6"]

        if is_root:
            print(f"[Install] Installing OpenGL libraries: {', '.join(packages)}")
            result = subprocess.run(
                ['apt-get', 'install', '-y'] + packages,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("[Install] ✓ OpenGL libraries installed successfully!")
                return True
            else:
                print(f"[Install] Warning: Failed to install OpenGL libraries")
                print(f"[Install] Error: {result.stderr}")
                print(f"[Install] You may need to run manually:")
                print(f"[Install]   sudo apt-get install {' '.join(packages)}")
                return True  # Don't fail installation, just warn
        else:
            print("[Install] Need sudo privileges to install system packages.")
            print(f"[Install] Please run:")
            print(f"[Install]   sudo apt-get install {' '.join(packages)}")
            print("[Install] Or run this installer with sudo:")
            print(f"[Install]   sudo python {__file__}")
            print("[Install] Continuing without installing system dependencies...")
            return True  # Don't fail installation

    except Exception as e:
        print(f"[Install] Warning: Could not install system dependencies: {e}")
        print(f"[Install] PyMeshLab remeshing may not work without OpenGL libraries.")
        print(f"[Install] To fix, run: sudo apt-get install libgl1 libglu1-mesa libglx-mesa0 libosmesa6")
        return True  # Don't fail installation, just warn


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
    # First, install system dependencies (Linux only)
    install_system_dependencies()

    # Then install Blender
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
