#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack Installer

Orchestrates modular install scripts for ComfyUI-GeometryPack.
Uses comfyui-envmanager for CUDA wheel resolution.
Blender is optional - run 'python blender_install.py' separately if needed.
"""

import sys
import subprocess

from install_scripts import (
    install_system_dependencies,
    install_python_dependencies,
)


def ensure_comfyui_envmanager():
    """Install comfyui-envmanager if not present."""
    try:
        import comfyui_envmanager
        return True
    except ImportError:
        print("[GeomPack] Installing comfyui-envmanager...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "comfyui-envmanager>=0.0.11"
            ])
            return True
        except subprocess.CalledProcessError as e:
            print(f"[GeomPack] Failed to install comfyui-envmanager: {e}")
            return False


def install_cuda_packages():
    """Install CUDA packages using comfyui-envmanager."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: CUDA Packages (via comfyui-envmanager)")
    print("="*60 + "\n")

    if not ensure_comfyui_envmanager():
        print("[GeomPack] Cannot install CUDA packages without comfyui-envmanager")
        return False

    try:
        from comfyui_envmanager import install, verify_installation
        from pathlib import Path

        # Install from comfyui_env.toml in this directory
        node_root = Path(__file__).parent.absolute()
        print(f"[GeomPack] Installing CUDA packages from {node_root}/comfyui_env.toml")

        install(config=node_root / "comfyui_env.toml")

        # Verify cumesh is importable
        try:
            import cumesh
            print("[GeomPack] CuMesh installed and verified!")
            return True
        except ImportError:
            print("[GeomPack] CuMesh installation completed but import failed")
            return False

    except Exception as e:
        print(f"[GeomPack] CUDA package installation failed: {e}")
        print("[GeomPack] CuMesh will not be available, but other features will work.")
        return False


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Installation")
    print("="*60 + "\n")
    print("This installer will set up:")
    print("  1. System dependencies (OpenGL libraries on Linux)")
    print("  2. Python dependencies (trimesh, pymeshlab, etc.)")
    print("  3. CUDA packages via comfyui-envmanager (cumesh)")
    print("")
    print("Note: Blender is optional. If you need Blender nodes")
    print("(UV Unwrap, Remesh, etc.), run: python blender_install.py")
    print("")

    results = {
        'system_deps': False,
        'python_deps': False,
        'cuda_packages': False,
    }

    # Install in order
    results['system_deps'] = install_system_dependencies()
    results['python_deps'] = install_python_dependencies()
    results['cuda_packages'] = install_cuda_packages()

    # Print summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'+ Success' if results['system_deps'] else 'x Failed'}")
    print(f"  Python Dependencies: {'+ Success' if results['python_deps'] else 'x Failed'}")
    print(f"  CUDA Packages:       {'+ Success' if results['cuda_packages'] else '~ Skipped'}")
    print("="*60 + "\n")

    if results['python_deps']:
        print("Installation completed successfully!")
        print("You can now use ComfyUI-GeometryPack nodes in ComfyUI.")
        print("")
        print("For Blender-based nodes (UV Unwrap, Remesh, etc.), run:")
        print("  python blender_install.py")
        print("")
        sys.exit(0)
    else:
        print("Installation completed with issues.")
        if not results['python_deps']:
            print("\nPython dependencies failed to install. You can:")
            print("  1. Try running: pip install -r requirements.txt")
            print("  2. Check your Python environment and permissions")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
