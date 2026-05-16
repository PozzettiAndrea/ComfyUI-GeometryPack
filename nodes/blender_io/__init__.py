# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Blender I/O nodes - requires bpy.

Each submodule is imported defensively: a broken bpy install (e.g. the
Windows tbb12.dll mismatch between bpy and trimesh[easy]'s embreex)
should disable only the affected nodes, not the rest of GeometryPack.
"""

import logging

log = logging.getLogger("geometrypack")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .load_mesh_blend import NODE_CLASS_MAPPINGS as LOAD_MESH_BLEND_MAPS, NODE_DISPLAY_NAME_MAPPINGS as LOAD_MESH_BLEND_DISP
    NODE_CLASS_MAPPINGS.update(LOAD_MESH_BLEND_MAPS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_MESH_BLEND_DISP)
except Exception as e:
    log.warning("blender_io.load_mesh_blend disabled: %s", e)

try:
    from .load_mesh_fbx import NODE_CLASS_MAPPINGS as LOAD_MESH_FBX_MAPS, NODE_DISPLAY_NAME_MAPPINGS as LOAD_MESH_FBX_DISP
    NODE_CLASS_MAPPINGS.update(LOAD_MESH_FBX_MAPS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_MESH_FBX_DISP)
except Exception as e:
    log.warning("blender_io.load_mesh_fbx disabled: %s", e)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
