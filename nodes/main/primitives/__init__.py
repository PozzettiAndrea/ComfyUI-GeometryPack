# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Primitive Nodes - Create basic geometric shapes
"""

from .create_primitive import NODE_CLASS_MAPPINGS as CreatePrimitive_mappings, NODE_DISPLAY_NAME_MAPPINGS as CreatePrimitive_display

# Aggregate all node mappings
NODE_CLASS_MAPPINGS = {
    **CreatePrimitive_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CreatePrimitive_display,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
