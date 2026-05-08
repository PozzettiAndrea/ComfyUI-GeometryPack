# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Distance computation module."""

from .mesh_to_mesh_distance import NODE_CLASS_MAPPINGS as MESH_DIST_MAPPINGS
from .mesh_to_mesh_distance import NODE_DISPLAY_NAME_MAPPINGS as MESH_DIST_DISPLAY
from .point_to_mesh_distance import NODE_CLASS_MAPPINGS as POINT_DIST_MAPPINGS
from .point_to_mesh_distance import NODE_DISPLAY_NAME_MAPPINGS as POINT_DIST_DISPLAY

# Also import from parent distance.py for other nodes (HausdorffDistance, ChamferDistance, ComputeSDF)

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **MESH_DIST_MAPPINGS,
    **POINT_DIST_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_DIST_DISPLAY,
    **POINT_DIST_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
