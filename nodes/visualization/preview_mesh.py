# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview mesh with interactive 3D viewer.

Displays mesh in an interactive Three.js viewer with orbit controls.
Allows rotating, panning, and zooming to inspect the mesh geometry.
"""

import logging

import trimesh as trimesh_module
import os
import tempfile
import uuid

from .mesh_helpers import is_point_cloud, get_face_count, get_geometry_type

log = logging.getLogger("geometrypack")

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None
from comfy_api.latest import io

class PreviewMeshNode(io.ComfyNode):
    """
    Preview mesh with interactive 3D viewer.

    Displays mesh in an interactive Three.js viewer with orbit controls.
    Allows rotating, panning, and zooming to inspect the mesh geometry.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMesh",
            display_name="Preview Mesh (Three.js)",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        """
        Export mesh to GLB and prepare for 3D preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        log.info("Preparing preview: %s - %d vertices, %d faces", get_geometry_type(trimesh), len(trimesh.vertices), get_face_count(trimesh))

        # Generate unique filename
        filename = f"preview_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to GLB (best format for Three.js)
        try:
            trimesh.export(filepath, file_type='glb')
            log.info("Exported to: %s", filepath)
        except Exception as e:
            log.error("Export failed: %s", e)
            # Fallback to OBJ
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            trimesh.export(filepath, file_type='obj')
            log.info("Exported to OBJ: %s", filepath)

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        return io.NodeOutput(ui={ "mesh_file": [filename], "vertex_count": [len(trimesh.vertices)], "face_count": [get_face_count(trimesh)], "bounds_min": [bounds[0].tolist()], "bounds_max": [bounds[1].tolist()], "extents": [extents.tolist()], "max_extent": [float(max_extent)], })

NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMesh": PreviewMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMesh": "Preview Mesh (Three.js)",
}
