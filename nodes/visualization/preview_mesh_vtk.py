# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview mesh with VTK.js scientific visualization viewer.

Displays mesh in an interactive VTK.js viewer with trackball controls.
Better for scientific visualization, mesh analysis, and large datasets.

Supports scalar field visualization: automatically detects vertex and face
attributes and exports to VTP format to preserve field data for visualization.
"""

import logging

import trimesh as trimesh_module
import os
import tempfile
import uuid

from .mesh_helpers import is_point_cloud, get_face_count, get_geometry_type
from ._vtp_export import export_mesh_with_scalars_vtp

log = logging.getLogger("geometrypack")

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None
from comfy_api.latest import io


class PreviewMeshVTKNode(io.ComfyNode):
    """
    Preview mesh with VTK.js scientific visualization viewer.

    Displays mesh in an interactive VTK.js viewer with trackball controls.
    Better for scientific visualization, mesh analysis, and large datasets.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshVTK",
            display_name="Preview Mesh",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Combo.Input("mode", options=["fields", "texture", "texture (PBR)"], default="fields"),
                io.Custom("TRIMESH").Input("trimesh", optional=True),
                io.Custom("VOXELGRID").Input("voxelgrid", optional=True),
            ],
        )

    @classmethod
    def execute(cls, mode="fields", trimesh=None, voxelgrid=None):
        """
        Export mesh and prepare for VTK.js preview.

        Supports two visualization modes:
        - fields: Scientific visualization with scalar fields and colormaps (VTP/STL export)
        - texture: Textured mesh visualization with materials (GLB export)

        Args:
            mode: Visualization mode - "fields" or "texture"
            trimesh: Input trimesh_module.Trimesh object (optional)
            voxelgrid: Input trimesh.VoxelGrid object (optional)

        Returns:
            dict: UI data for frontend widget
        """
        # Use whichever input is provided
        if trimesh is None and voxelgrid is None:
            raise ValueError("Either trimesh or voxelgrid input is required")

        mesh_input = trimesh if trimesh is not None else voxelgrid

        # Handle VOXEL_GRID input (trimesh.VoxelGrid from MeshToVoxel node)
        if hasattr(mesh_input, 'as_boxes'):  # It's a trimesh.VoxelGrid
            voxel_shape = mesh_input.matrix.shape
            trimesh = mesh_input.as_boxes()
            log.info("Converted voxel grid %s to box mesh: %d vertices", voxel_shape, len(trimesh.vertices))
        else:
            trimesh = mesh_input

        log.info("Preparing preview: %s - %d vertices, %d faces", get_geometry_type(trimesh), len(trimesh.vertices), get_face_count(trimesh))

        # Check for scalar fields (vertex/face attributes)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0
        has_face_attrs = hasattr(trimesh, 'face_attributes') and len(trimesh.face_attributes) > 0
        has_fields = has_vertex_attrs or has_face_attrs

        log.debug("hasattr vertex_attributes: %s", hasattr(trimesh, 'vertex_attributes'))
        log.debug("hasattr face_attributes: %s", hasattr(trimesh, 'face_attributes'))
        if hasattr(trimesh, 'vertex_attributes'):
            log.debug("vertex_attributes: %s", trimesh.vertex_attributes)
            log.debug("len(vertex_attributes): %d", len(trimesh.vertex_attributes))
        if hasattr(trimesh, 'face_attributes'):
            log.debug("face_attributes: %s", trimesh.face_attributes)
            log.debug("len(face_attributes): %d", len(trimesh.face_attributes))
        log.debug("has_vertex_attrs: %s", has_vertex_attrs)
        log.debug("has_face_attrs: %s", has_face_attrs)
        log.debug("has_fields: %s", has_fields)

        # Check for visual data (textures/vertex colors)
        has_visual = hasattr(trimesh, 'visual') and trimesh.visual is not None
        visual_kind = trimesh.visual.kind if has_visual else None
        has_texture = visual_kind == 'texture' and hasattr(trimesh.visual, 'material') if has_visual else False
        has_vertex_colors = visual_kind == 'vertex' if has_visual else False
        has_material = has_texture

        log.info("Mode: %s", mode)
        log.info("Visual data - has_visual: %s, kind: %s, texture: %s, vertex_colors: %s", has_visual, visual_kind, has_texture, has_vertex_colors)

        # Check if this is a point cloud
        is_pc = is_point_cloud(trimesh)

        # Choose export format based on visualization mode
        if mode == "texture (PBR)":
            # PBR mode: Export GLB and use Three.js PBR viewer
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.glb"
            viewer_type = "pbr"
            log.info("Using PBR mode - GLB export with Three.js PBR viewer")
        elif mode == "texture":
            # Texture mode: Export GLB to preserve textures/materials/UVs
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.glb"
            viewer_type = "texture"
            log.info("Using texture mode - GLB export")
        else:
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.vtp"
            viewer_type = "fields"
            log.info("Using fields mode - VTP export")

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Set alpha mode based on viewing mode
        if mode == "texture":
            # Texture mode: force opaque for solid color viewing (RGB only)
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                if hasattr(trimesh.visual.material, 'alphaMode'):
                    trimesh.visual.material.alphaMode = 'OPAQUE'
                    log.info("Set alphaMode to OPAQUE for texture mode")
        elif mode == "texture (PBR)":
            # PBR mode: use BLEND for transparency support
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                if hasattr(trimesh.visual.material, 'alphaMode'):
                    trimesh.visual.material.alphaMode = 'BLEND'
                    log.info("Set alphaMode to BLEND for PBR mode")

        # Export mesh
        try:
            if mode in ("texture", "texture (PBR)"):
                # Export GLB for texture/PBR rendering
                trimesh.export(filepath, file_type='glb', include_normals=True)
                log.info("Exported GLB to: %s", filepath)
            else:
                export_mesh_with_scalars_vtp(trimesh, filepath)
                log.info("Exported VTP to: %s", filepath)
        except Exception as e:
            log.error("Export failed: %s", e)
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            log.info("Exported to OBJ: %s", filepath)

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents

        # Handle case where extents/bounds are None (can happen with certain mesh configurations)
        if extents is None or bounds is None:
            import numpy as np
            vertices_arr = np.asarray(trimesh.vertices)
            if len(vertices_arr) > 0:
                bounds = np.array([vertices_arr.min(axis=0), vertices_arr.max(axis=0)])
                extents = bounds[1] - bounds[0]
            else:
                # Empty mesh - use default bounds
                bounds = np.array([[0, 0, 0], [1, 1, 1]])
                extents = np.array([1, 1, 1])

        max_extent = max(extents)

        # Check if mesh is watertight (only for actual meshes, not point clouds)
        is_watertight = False if is_point_cloud(trimesh) else trimesh.is_watertight

        # Calculate volume and area (only for meshes with faces, not point clouds)
        volume = None
        area = None
        if not is_point_cloud(trimesh):
            try:
                if is_watertight:
                    volume = float(trimesh.volume)
                area = float(trimesh.area)
            except Exception as e:
                log.info("Could not calculate volume/area: %s", e)

        # Get field names (vertex/face data arrays) - for field visualization UI
        field_names = []
        if has_vertex_attrs:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
            log.info("Vertex attributes: %s", list(trimesh.vertex_attributes.keys()))
        if has_face_attrs:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])
            log.info("Face attributes: %s", list(trimesh.face_attributes.keys()))

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "viewer_type": [viewer_type],  # "fields" or "texture" - tells frontend which viewer to load
            "mode": [mode],  # User-selected mode
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [get_face_count(trimesh)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
        }

        # Add mode-specific metadata
        if viewer_type in ("texture", "pbr"):
            # Texture/PBR mode metadata
            ui_data.update({
                "has_texture": [has_texture],
                "has_vertex_colors": [has_vertex_colors],
                "has_material": [has_material],
                "visual_kind": [visual_kind if visual_kind else "none"],
            })
        else:
            # Fields mode metadata
            ui_data["field_names"] = [field_names]  # Field visualization data

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]

        if viewer_type == "pbr":
            log.info("PBR mode info: watertight=%s, volume=%s, area=%s, texture=%s, vertex_colors=%s", is_watertight, volume, area, has_texture, has_vertex_colors)
        elif viewer_type == "texture":
            log.info("Texture mode info: watertight=%s, volume=%s, area=%s, texture=%s, vertex_colors=%s", is_watertight, volume, area, has_texture, has_vertex_colors)
        elif field_names:
            log.info("Fields mode info: watertight=%s, volume=%s, area=%s, fields=%s", is_watertight, volume, area, field_names)
        else:
            log.info("Fields mode info: watertight=%s, volume=%s, area=%s, no fields", is_watertight, volume, area)

        return io.NodeOutput(ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTK": "Preview Mesh",
}
