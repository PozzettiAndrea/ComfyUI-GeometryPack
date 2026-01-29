# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Reconstruction nodes (main dependencies only - no CGAL Alpha Wrap)
"""

from .reconstruct_surface import NODE_CLASS_MAPPINGS as RECONSTRUCT_MAPS, NODE_DISPLAY_NAME_MAPPINGS as RECONSTRUCT_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(RECONSTRUCT_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(RECONSTRUCT_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
