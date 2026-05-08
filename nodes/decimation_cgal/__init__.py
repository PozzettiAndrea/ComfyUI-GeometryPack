# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL decimation backend nodes.
"""

from .cgal_edge_collapse import NODE_CLASS_MAPPINGS as CGAL_MAPS, NODE_DISPLAY_NAME_MAPPINGS as CGAL_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(CGAL_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(CGAL_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
