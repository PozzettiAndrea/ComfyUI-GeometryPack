# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CuMesh Installation - GPU-accelerated mesh processing library.

Tries to install cumesh via:
1. Pre-built wheel from cumesh-wheels repository
2. Source compilation from JeffreyXiang/CuMesh

Requires CUDA and PyTorch. Gracefully skips if not available.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path


# Known wheel configurations
WHEEL_REPO_BASE = "https://github.com/PozzettiAndrea/cumesh-wheels/releases/download"
SOURCE_REPO = "https://github.com/JeffreyXiang/CuMesh.git"


def _check_cuda_available():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.version.cuda, torch.__version__
        else:
            return False, None, torch.__version__
    except ImportError:
        return False, None, None


def _get_python_version():
    """Get Python version string for wheel filename."""
    return f"{sys.version_info.major}{sys.version_info.minor}"


def _verify_cumesh_import():
    """Try to import cumesh and verify it works."""
    try:
        import cumesh
        # Try to instantiate the main class
        _ = cumesh.CuMesh
        return True
    except Exception:
        return False


def _install_wheel(cuda_version, torch_version, python_version):
    """Try to install cumesh from pre-built wheel."""
    # Format versions for wheel URL
    # CUDA: "12.8" -> "128", Torch: "2.8.0+cu128" -> "280", "28" (major.minor)
    cuda_short = cuda_version.replace(".", "")[:3] if cuda_version else None
    torch_clean = torch_version.split("+")[0] if torch_version else None
    torch_short = torch_clean.replace(".", "")[:3] if torch_clean else None
    torch_mm = "".join(torch_clean.split(".")[:2]) if torch_clean else None  # "2.8.0" -> "28"

    if not cuda_short or not torch_short:
        print("[CuMesh] Cannot determine CUDA/PyTorch versions for wheel selection")
        return False

    # Construct wheel URL - filename includes build tag like +cu128torch28
    wheel_tag = f"cu{cuda_short}-torch{torch_short}"
    build_tag = f"cu{cuda_short}torch{torch_mm}"  # cu128torch28
    wheel_name = f"cumesh-0.0.1%2B{build_tag}-cp{python_version}-cp{python_version}-linux_x86_64.whl"
    wheel_url = f"{WHEEL_REPO_BASE}/{wheel_tag}/{wheel_name}"

    print(f"[CuMesh] Trying wheel: {wheel_tag}/{wheel_name}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", wheel_url],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"[CuMesh] Wheel installed successfully!")
            return True
        else:
            print(f"[CuMesh] Wheel installation failed (may not exist for this configuration)")
            if "404" in result.stderr or "not found" in result.stderr.lower():
                print(f"[CuMesh] Wheel not available for {wheel_tag}")
            return False

    except subprocess.TimeoutExpired:
        print("[CuMesh] Wheel download timed out")
        return False
    except Exception as e:
        print(f"[CuMesh] Wheel installation error: {e}")
        return False


def _install_from_source():
    """Try to install cumesh from source."""
    print("[CuMesh] Attempting source compilation...")
    print("[CuMesh] This may take several minutes...")

    # Create temporary directory for clone
    temp_dir = tempfile.mkdtemp(prefix="cumesh_build_")
    clone_path = Path(temp_dir) / "CuMesh"

    try:
        # Clone repository
        print(f"[CuMesh] Cloning {SOURCE_REPO}...")
        result = subprocess.run(
            ["git", "clone", "--recursive", SOURCE_REPO, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"[CuMesh] Git clone failed: {result.stderr}")
            return False

        # Install with pip
        print("[CuMesh] Building and installing (this may take a while)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
            cwd=str(clone_path),
            capture_output=True,
            text=True,
            timeout=1200  # 20 minutes for compilation
        )

        if result.returncode == 0:
            print("[CuMesh] Source build completed successfully!")
            return True
        else:
            print(f"[CuMesh] Source build failed:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("[CuMesh] Source build timed out (>20 minutes)")
        return False
    except FileNotFoundError:
        print("[CuMesh] Git not found - cannot clone repository")
        return False
    except Exception as e:
        print(f"[CuMesh] Source build error: {e}")
        return False
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def install_cumesh():
    """
    Install CuMesh library for GPU-accelerated mesh processing.

    Returns:
        bool: True if installed successfully or already available,
              False if installation failed (graceful failure).
    """
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: CuMesh Installation (Optional)")
    print("="*60 + "\n")

    # Check if already installed
    if _verify_cumesh_import():
        print("[CuMesh] Already installed and working!")
        return True

    # Check CUDA availability
    cuda_available, cuda_version, torch_version = _check_cuda_available()

    if not cuda_available:
        print("[CuMesh] CUDA not available - skipping cumesh installation")
        print("[CuMesh] This is fine! The 'cumesh' UV unwrap method will be disabled,")
        print("[CuMesh] but other UV unwrap methods (xatlas, libigl, blender) will work.")
        return True  # Graceful skip, not a failure

    print(f"[CuMesh] CUDA {cuda_version} detected with PyTorch {torch_version}")

    python_version = _get_python_version()
    print(f"[CuMesh] Python version: {sys.version_info.major}.{sys.version_info.minor}")

    # Try wheel installation
    print("\n[CuMesh] Trying pre-built wheel...")
    if _install_wheel(cuda_version, torch_version, python_version):
        if _verify_cumesh_import():
            print("[CuMesh] Installation verified!")
            return True
        else:
            print("[CuMesh] Wheel installed but import failed")

    # Wheel not available or failed
    print("\n" + "-"*60)
    print("[CuMesh] WARNING: CuMesh installation failed")
    print("[CuMesh] The 'cumesh' UV unwrap method will not be available.")
    print("[CuMesh] Other UV unwrap methods (xatlas, libigl, blender) will still work.")
    print("-"*60)

    # Return True for graceful failure - don't break the whole install
    return True
