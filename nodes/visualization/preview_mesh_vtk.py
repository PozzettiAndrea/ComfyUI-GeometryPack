# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview mesh with VTK.js scientific visualization viewer.

Displays mesh in an interactive VTK.js viewer with trackball controls.
Better for scientific visualization, mesh analysis, and large datasets.

Supports scalar field visualization: automatically detects vertex and face
attributes and exports to VTP format to preserve field data for visualization.
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid

from .._utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type
from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


class PreviewMeshVTKNode:
    """
    Preview mesh with VTK.js scientific visualization viewer.

    Displays mesh in an interactive VTK.js viewer with trackball controls.
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["fields", "texture", "texture (PBR)"], {"default": "fields"}),
            },
            "optional": {
                "trimesh": ("TRIMESH",),
                "voxelgrid": ("VOXELGRID",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk(self, mode="fields", trimesh=None, voxelgrid=None):
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
            print(f"[PreviewMeshVTK] Converted voxel grid {voxel_shape} to box mesh: {len(trimesh.vertices)} vertices")
        else:
            trimesh = mesh_input

        print(f"[PreviewMeshVTK] Preparing preview: {get_geometry_type(trimesh)} - {len(trimesh.vertices)} vertices, {get_face_count(trimesh)} faces")

        # Check for scalar fields (vertex/face attributes)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0
        has_face_attrs = hasattr(trimesh, 'face_attributes') and len(trimesh.face_attributes) > 0
        has_fields = has_vertex_attrs or has_face_attrs

        print(f"[PreviewMeshVTK] DEBUG - hasattr vertex_attributes: {hasattr(trimesh, 'vertex_attributes')}")
        print(f"[PreviewMeshVTK] DEBUG - hasattr face_attributes: {hasattr(trimesh, 'face_attributes')}")
        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTK] DEBUG - vertex_attributes: {trimesh.vertex_attributes}")
            print(f"[PreviewMeshVTK] DEBUG - len(vertex_attributes): {len(trimesh.vertex_attributes)}")
        if hasattr(trimesh, 'face_attributes'):
            print(f"[PreviewMeshVTK] DEBUG - face_attributes: {trimesh.face_attributes}")
            print(f"[PreviewMeshVTK] DEBUG - len(face_attributes): {len(trimesh.face_attributes)}")
        print(f"[PreviewMeshVTK] DEBUG - has_vertex_attrs: {has_vertex_attrs}")
        print(f"[PreviewMeshVTK] DEBUG - has_face_attrs: {has_face_attrs}")
        print(f"[PreviewMeshVTK] DEBUG - has_fields: {has_fields}")

        # Check for visual data (textures/vertex colors)
        has_visual = hasattr(trimesh, 'visual') and trimesh.visual is not None
        visual_kind = trimesh.visual.kind if has_visual else None
        has_texture = visual_kind == 'texture' and hasattr(trimesh.visual, 'material') if has_visual else False
        has_vertex_colors = visual_kind == 'vertex' if has_visual else False
        has_material = has_texture

        print(f"[PreviewMeshVTK] Mode: {mode}")
        print(f"[PreviewMeshVTK] Visual data - has_visual: {has_visual}, kind: {visual_kind}, texture: {has_texture}, vertex_colors: {has_vertex_colors}")

        # Check if this is a point cloud
        is_pc = is_point_cloud(trimesh)

        # Choose export format based on visualization mode
        if mode == "texture (PBR)":
            # PBR mode: Export GLB and use Three.js PBR viewer
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.glb"
            viewer_type = "pbr"
            print(f"[PreviewMeshVTK] Using PBR mode - GLB export with Three.js PBR viewer")
        elif mode == "texture":
            # Texture mode: Export GLB to preserve textures/materials/UVs
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.glb"
            viewer_type = "texture"
            print(f"[PreviewMeshVTK] Using texture mode - GLB export")
        else:
            # Fields mode: Export VTP/STL for scalar field visualization
            if has_fields or is_pc:
                # Export to VTP for: scalar fields OR point clouds (STL doesn't support point clouds)
                filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.vtp"
                print(f"[PreviewMeshVTK] Using VTP format (fields={has_fields}, point_cloud={is_pc})")
            else:
                # Export to STL (compact format for simple surface meshes)
                filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.stl"
            viewer_type = "fields"
            print(f"[PreviewMeshVTK] Using fields mode - VTP/STL export")

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
                    print(f"[PreviewMeshVTK] Set alphaMode to OPAQUE for texture mode")
        elif mode == "texture (PBR)":
            # PBR mode: use BLEND for transparency support
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                if hasattr(trimesh.visual.material, 'alphaMode'):
                    trimesh.visual.material.alphaMode = 'BLEND'
                    print(f"[PreviewMeshVTK] Set alphaMode to BLEND for PBR mode")

        # Export mesh
        try:
            if mode in ("texture", "texture (PBR)"):
                # Export GLB for texture/PBR rendering
                trimesh.export(filepath, file_type='glb', include_normals=True)
                print(f"[PreviewMeshVTK] Exported GLB to: {filepath}")
            elif has_fields or is_pc:
                # Use VTP exporter for fields or point clouds
                export_mesh_with_scalars_vtp(trimesh, filepath)
                print(f"[PreviewMeshVTK] Exported VTP to: {filepath}")
            else:
                # Use STL for simple surface meshes
                trimesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshVTK] Exported STL to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTK] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTK] Exported to OBJ: {filepath}")

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
                print(f"[PreviewMeshVTK] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays) - for field visualization UI
        field_names = []
        if has_vertex_attrs:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
            print(f"[PreviewMeshVTK] Vertex attributes: {list(trimesh.vertex_attributes.keys())}")
        if has_face_attrs:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])
            print(f"[PreviewMeshVTK] Face attributes: {list(trimesh.face_attributes.keys())}")

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
            print(f"[PreviewMeshVTK] PBR mode info: watertight={is_watertight}, volume={volume}, area={area}, texture={has_texture}, vertex_colors={has_vertex_colors}")
        elif viewer_type == "texture":
            print(f"[PreviewMeshVTK] Texture mode info: watertight={is_watertight}, volume={volume}, area={area}, texture={has_texture}, vertex_colors={has_vertex_colors}")
        elif field_names:
            print(f"[PreviewMeshVTK] Fields mode info: watertight={is_watertight}, volume={volume}, area={area}, fields={field_names}")
        else:
            print(f"[PreviewMeshVTK] Fields mode info: watertight={is_watertight}, volume={volume}, area={area}, no fields")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTK": "Preview Mesh",
}
