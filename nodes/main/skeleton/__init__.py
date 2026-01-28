# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Skeleton extraction and visualization module."""

from .extract_skeleton import NODE_CLASS_MAPPINGS as EXTRACT_MAPPINGS
from .extract_skeleton import NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY
from .mesh_from_skeleton import NODE_CLASS_MAPPINGS as MESH_MAPPINGS
from .mesh_from_skeleton import NODE_DISPLAY_NAME_MAPPINGS as MESH_DISPLAY

# Also import from parent skeleton.py for other nodes (ExtractSkeleton, SkeletonToTrimesh, SkeletonToMesh)

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **EXTRACT_MAPPINGS,
    **MESH_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **EXTRACT_DISPLAY,
    **MESH_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
