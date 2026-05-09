# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Individual boolean backend nodes (hidden from UI, invoked via unified Boolean node).
"""

from .libigl_cgal import NODE_CLASS_MAPPINGS as CGAL_MAPS, NODE_DISPLAY_NAME_MAPPINGS as CGAL_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(CGAL_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(CGAL_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
