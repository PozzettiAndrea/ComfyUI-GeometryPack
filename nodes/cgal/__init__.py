# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL-dependent nodes - requires igl.copyleft.cgal or CGAL Python bindings.
These nodes will only load if CGAL dependencies are available.
"""

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
