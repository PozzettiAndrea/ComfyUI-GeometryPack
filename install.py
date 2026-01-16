#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack Installer

Orchestrates modular install scripts for ComfyUI-GeometryPack.
Uses comfy-env for isolated Python environment with bpy + cumesh.

For full functionality (Blender operations, cumesh GPU acceleration):
    cd custom_nodes/ComfyUI-GeometryPack
    comfy-env install
"""

import sys

from install_scripts import (
    install_system_dependencies,
    install_python_dependencies,
)


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Installation")
    print("="*60 + "\n")
    print("This installer will set up:")
    print("  1. System dependencies (OpenGL libraries on Linux)")
    print("  2. Python dependencies (trimesh, pymeshlab, etc.)")
    print("")
    print("For Blender operations (UV Unwrap, Remesh, Boolean, etc.)")
    print("and GPU-accelerated cumesh, run comfy-env install:")
    print("")
    print("    cd custom_nodes/ComfyUI-GeometryPack")
    print("    comfy-env install")
    print("")
    print("This creates an isolated Python 3.11 environment with bpy + cumesh.")
    print("")

    results = {
        'system_deps': False,
        'python_deps': False,
    }

    # Install in order
    results['system_deps'] = install_system_dependencies()
    results['python_deps'] = install_python_dependencies()

    # Print summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'+ Success' if results['system_deps'] else 'x Failed'}")
    print(f"  Python Dependencies: {'+ Success' if results['python_deps'] else 'x Failed'}")
    print("="*60 + "\n")

    if results['python_deps']:
        print("Basic installation completed successfully!")
        print("You can now use basic ComfyUI-GeometryPack nodes in ComfyUI.")
        print("")
        print("For full functionality (Blender ops, GPU cumesh), run:")
        print("")
        print("    cd custom_nodes/ComfyUI-GeometryPack")
        print("    comfy-env install")
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
