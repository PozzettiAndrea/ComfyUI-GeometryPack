#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack Installer

Uses comfy-env for isolated Python environment with bpy + cumesh.
"""

import sys
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-GeometryPack Installation")
    print("=" * 60)

    from comfy_env import install, IsolatedEnvManager, discover_config

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install
    try:
        install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
    except Exception as e:
        print(f"\n[UniRig] Installation FAILED: {e}")
        print("[GeometryPack] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-GeometryPack/issues")
        return 1

    print("\n" + "=" * 60)
    print("[GeometryPack] Installation completed!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())