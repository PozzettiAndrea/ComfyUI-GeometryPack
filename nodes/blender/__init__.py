# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Blender-dependent nodes - requires bpy (Blender Python module).
These nodes will only load if bpy is available.
"""

# Check for bpy availability at import time
def _check_bpy_available():
    """Check if bpy (Blender Python module) is available."""
    try:
        # Try to import bpy via the setup function
        from .._utils import setup_bpy_dll_path
        setup_bpy_dll_path()
        import bpy
        return True
    except (ImportError, Exception):
        pass

    # Direct import attempt
    try:
        import bpy
        return True
    except ImportError:
        pass

    return False


if not _check_bpy_available():
    raise ImportError(
        "Blender bpy module not available. "
        "Install with: pip install bpy"
    )

# Import submodules
from . import io
from . import boolean
from . import remeshing
from . import texture_remeshing

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(io.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(boolean.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(texture_remeshing.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(io.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(boolean.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(texture_remeshing.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
