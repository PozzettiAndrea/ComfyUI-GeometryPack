# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
BPY Worker Manager - Manages the isolated bpy environment worker.

Provides a simple interface for calling bpy_bridge functions in the
isolated Python 3.11 environment with bpy installed.

Usage:
    from .._utils.bpy_worker import call_bpy
    result = call_bpy('bpy_smart_uv_project', vertices=v, faces=f, ...)
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np

# Lazy-loaded worker
_worker = None


def _get_worker():
    """Get or create the VenvWorker for the geometrypack environment."""
    global _worker
    if _worker is not None:
        return _worker

    from comfy_env import VenvWorker

    # Find the isolated environment
    node_dir = Path(__file__).parent.parent.parent  # ComfyUI-GeometryPack/
    env_path = node_dir / "_env_geometrypack"

    if not env_path.exists():
        raise RuntimeError(
            f"Isolated environment not found at {env_path}\n"
            "Run 'comfy-env install --isolated' to create it."
        )

    python_path = env_path / "bin" / "python"
    if not python_path.exists():
        python_path = env_path / "Scripts" / "python.exe"  # Windows

    if not python_path.exists():
        raise RuntimeError(f"Python not found in isolated environment: {env_path}")

    # Create worker with sys.path including our modules
    utils_dir = Path(__file__).parent
    _worker = VenvWorker(
        python=str(python_path),
        sys_path=[str(utils_dir)],
        name="geometrypack"
    )

    return _worker


def call_bpy(func_name: str, **kwargs) -> Dict[str, Any]:
    """
    Call a function in bpy_bridge via the isolated worker.

    Args:
        func_name: Name of function in bpy_bridge module
        **kwargs: Arguments to pass to the function

    Returns:
        Result dict from the function

    Example:
        result = call_bpy('bpy_smart_uv_project',
            vertices=[[0,0,0], [1,0,0], [0,1,0]],
            faces=[[0,1,2]],
            angle_limit=1.15,
            island_margin=0.02,
            scale_to_bounds=True
        )
    """
    worker = _get_worker()

    # Convert numpy arrays to lists for IPC
    converted_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            converted_kwargs[key] = value.tolist()
        else:
            converted_kwargs[key] = value

    return worker.call_module(
        module='bpy_bridge',
        func=func_name,
        **converted_kwargs
    )


def shutdown():
    """Shutdown the worker process."""
    global _worker
    if _worker is not None:
        _worker.shutdown()
        _worker = None
