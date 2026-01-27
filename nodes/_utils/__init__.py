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
    import sys as _sys
    print(f"[setup_bpy_dll_path] platform={sys.platform}", file=_sys.stderr, flush=True)
    if sys.platform == "win32":
        try:
            import importlib.util
            spec = importlib.util.find_spec("bpy")
            print(f"[setup_bpy_dll_path] spec={spec}", file=_sys.stderr, flush=True)
            if spec and spec.origin:
                bpy_dir = os.path.dirname(spec.origin)
                print(f"[setup_bpy_dll_path] bpy_dir={bpy_dir}", file=_sys.stderr, flush=True)
                # List DLLs in that directory
                dll_count = len([f for f in os.listdir(bpy_dir) if f.endswith('.dll')])
                print(f"[setup_bpy_dll_path] found {dll_count} DLLs in bpy_dir", file=_sys.stderr, flush=True)
                os.add_dll_directory(bpy_dir)
                print(f"[setup_bpy_dll_path] added DLL directory successfully", file=_sys.stderr, flush=True)
            else:
                print(f"[setup_bpy_dll_path] spec.origin is None or spec is None", file=_sys.stderr, flush=True)
        except Exception as e:
            print(f"[setup_bpy_dll_path] ERROR: {e}", file=_sys.stderr, flush=True)


# Re-export commonly used utilities for convenience
from .mesh_ops import *
from .blender_bridge import *

__all__ = ['mesh_ops', 'blender_bridge', 'setup_bpy_dll_path']
