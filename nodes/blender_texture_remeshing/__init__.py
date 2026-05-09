# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Blender texture remeshing nodes - requires bpy
"""

from .remesh_uv import NODE_CLASS_MAPPINGS as REMESH_UV_MAPS, NODE_DISPLAY_NAME_MAPPINGS as REMESH_UV_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(REMESH_UV_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(REMESH_UV_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
