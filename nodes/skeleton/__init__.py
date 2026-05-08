# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Skeleton extraction and visualization module."""

from .extract_skeleton import NODE_CLASS_MAPPINGS as EXTRACT_MAPPINGS
from .extract_skeleton import NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY
from .backends import NODE_CLASS_MAPPINGS as BACKENDS_MAPPINGS
from .backends import NODE_DISPLAY_NAME_MAPPINGS as BACKENDS_DISPLAY
from .mesh_from_skeleton import NODE_CLASS_MAPPINGS as MESH_MAPPINGS
from .mesh_from_skeleton import NODE_DISPLAY_NAME_MAPPINGS as MESH_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **EXTRACT_MAPPINGS,
    **BACKENDS_MAPPINGS,
    **MESH_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **EXTRACT_DISPLAY,
    **BACKENDS_DISPLAY,
    **MESH_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
