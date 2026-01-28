# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Internal utility modules for GeometryPack nodes.

This package contains shared utility functions used across multiple nodes.
These are internal implementation details and should not be imported directly by users.
"""

import os
import sys

def setup_bpy_dll_path():
    """Add bpy's DLL directory to the search path (required for Python 3.8+ on Windows).

    Python 3.8+ changed DLL loading - it no longer uses PATH. The bpy package has
    70+ DLLs that need to be found when importing. This function adds the bpy
    package directory to the DLL search path.

    Must be called BEFORE `import bpy`.
    """
    if sys.platform == "win32":
        try:
            import importlib.util
            spec = importlib.util.find_spec("bpy")
            if spec and spec.origin:
                bpy_dir = os.path.dirname(spec.origin)
                os.add_dll_directory(bpy_dir)
        except Exception:
            pass  # Best effort


# Re-export commonly used utilities for convenience
from .mesh_ops import *

__all__ = ['mesh_ops', 'setup_bpy_dll_path']
