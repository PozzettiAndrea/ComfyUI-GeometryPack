# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL reconstruction nodes - requires pymeshlab with CGAL support
"""

from .alpha_wrap import NODE_CLASS_MAPPINGS as ALPHA_WRAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as ALPHA_WRAP_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(ALPHA_WRAP_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(ALPHA_WRAP_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
