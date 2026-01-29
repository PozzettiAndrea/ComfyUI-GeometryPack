# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
UV nodes aggregation.

The unified UV Unwrap node consolidates all UV unwrapping methods into a single node
with dynamic parameter exposure based on the selected method.
"""

# Unified UV unwrap node
from .uv_unwrap import UVUnwrapNode, NODE_CLASS_MAPPINGS as UV_UNWRAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as UV_UNWRAP_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register only the unified node
NODE_CLASS_MAPPINGS.update(UV_UNWRAP_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(UV_UNWRAP_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
