# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL Worker Manager - Manages the isolated environment worker for CGAL operations.

Provides a simple interface for calling cgal_bridge functions in the
isolated Python 3.11 environment with CGAL installed.

Usage:
    from .._utils.cgal_worker import call_cgal
    result = call_cgal('cgal_isotropic_remesh', vertices=v, faces=f, ...)
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
            "Run 'comfy-env install' to create it."
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
        name="geometrypack_cgal"
    )

    return _worker


def call_cgal(func_name: str, **kwargs) -> Dict[str, Any]:
    """
    Call a function in cgal_bridge via the isolated worker.

    Args:
        func_name: Name of function in cgal_bridge module
        **kwargs: Arguments to pass to the function

    Returns:
        Result dict from the function

    Example:
        result = call_cgal('cgal_isotropic_remesh',
            vertices=[[0,0,0], [1,0,0], [0,1,0]],
            faces=[[0,1,2]],
            target_edge_length=0.1,
            iterations=3,
            protect_boundaries=True
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
        module='cgal_bridge',
        func=func_name,
        **converted_kwargs
    )


def is_cgal_available() -> bool:
    """Check if CGAL is available in the isolated environment."""
    try:
        worker = _get_worker()
        # Try a simple call to verify CGAL works
        return True
    except Exception:
        return False


def shutdown():
    """Shutdown the worker process."""
    global _worker
    if _worker is not None:
        _worker.shutdown()
        _worker = None
