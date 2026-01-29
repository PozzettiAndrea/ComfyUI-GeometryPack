# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL-dependent nodes - requires igl.copyleft.cgal or CGAL Python bindings.
These nodes will only load if CGAL dependencies are available.
"""

# Check for CGAL availability at import time
def _check_cgal_available():
    """Check if any CGAL dependency is available."""
    # Try libigl with CGAL
    try:
        import igl.copyleft.cgal
        return True
    except ImportError:
        pass

    # Try direct CGAL Python bindings
    try:
        from CGAL import CGAL_Polygon_mesh_processing
        return True
    except ImportError:
        pass

    # Try pymeshlab (for alpha_wrap which uses CGAL internally)
    try:
        import pymeshlab
        return True
    except ImportError:
        pass

    return False


if not _check_cgal_available():
    raise ImportError(
        "CGAL dependencies not available. "
        "Install with: pip install libigl cgal pymeshlab"
    )

# Import submodules
from . import repair
from . import boolean
from . import remeshing
from . import reconstruction

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(repair.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(boolean.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(reconstruction.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(repair.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(boolean.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(reconstruction.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
