# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
UV nodes aggregation.

The unified UV Unwrap node consolidates all UV unwrapping methods into a single node
with dynamic parameter exposure based on the selected method.
"""

# Unified UV unwrap node (frontend)
from .uv_unwrap import NODE_CLASS_MAPPINGS as UV_UNWRAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as UV_UNWRAP_DISP

# Main-env backend nodes
from .backends import NODE_CLASS_MAPPINGS as BACKENDS_MAPS, NODE_DISPLAY_NAME_MAPPINGS as BACKENDS_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(UV_UNWRAP_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(UV_UNWRAP_DISP)
NODE_CLASS_MAPPINGS.update(BACKENDS_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(BACKENDS_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
