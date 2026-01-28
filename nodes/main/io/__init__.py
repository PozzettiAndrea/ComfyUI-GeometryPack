# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
I/O Nodes - Load and save mesh files (main dependencies only)
"""

from .load_mesh import NODE_CLASS_MAPPINGS as LoadMesh_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMesh_display
from .load_mesh_path import NODE_CLASS_MAPPINGS as LoadMeshPath_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshPath_display
from .load_mesh_batch import NODE_CLASS_MAPPINGS as LoadMeshBatch_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshBatch_display
from .load_mesh_glob import NODE_CLASS_MAPPINGS as LoadMeshGlob_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshGlob_display
from .load_mesh_fbx import NODE_CLASS_MAPPINGS as LoadMeshFBX_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshFBX_display
from .save_mesh import NODE_CLASS_MAPPINGS as SaveMesh_mappings, NODE_DISPLAY_NAME_MAPPINGS as SaveMesh_display
from .save_mesh_batch import NODE_CLASS_MAPPINGS as SaveMeshBatch_mappings, NODE_DISPLAY_NAME_MAPPINGS as SaveMeshBatch_display
from .get_mesh_filename import NODE_CLASS_MAPPINGS as GetMeshFilename_mappings, NODE_DISPLAY_NAME_MAPPINGS as GetMeshFilename_display

NODE_CLASS_MAPPINGS = {
    **LoadMesh_mappings,
    **LoadMeshPath_mappings,
    **LoadMeshBatch_mappings,
    **LoadMeshGlob_mappings,
    **LoadMeshFBX_mappings,
    **SaveMesh_mappings,
    **SaveMeshBatch_mappings,
    **GetMeshFilename_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LoadMesh_display,
    **LoadMeshPath_display,
    **LoadMeshBatch_display,
    **LoadMeshGlob_display,
    **LoadMeshFBX_display,
    **SaveMesh_display,
    **SaveMeshBatch_display,
    **GetMeshFilename_display,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
