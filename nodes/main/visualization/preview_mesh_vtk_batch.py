# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview batch of meshes with VTK.js scientific visualization viewer with index navigation.

Displays meshes from a batch in an interactive VTK.js viewer with trackball controls.
Includes navigation buttons to cycle through meshes in the batch.
Better for scientific visualization, mesh analysis, and large datasets.

Supports scalar field visualization: automatically detects vertex and face
attributes and exports to VTP format to preserve field data for visualization.
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid

from ..._utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type
from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


class PreviewMeshVTKBatchNode:
    """
    Preview batch of meshes with VTK.js scientific visualization viewer with index navigation.

    Displays meshes from a batch in an interactive VTK.js viewer with trackball controls.
    Includes navigation buttons to cycle through meshes in the batch.
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "mode": (["fields", "texture"], {"default": "fields"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk_batch"
    CATEGORY = "geompack/visualization"
    INPUT_IS_LIST = True

    def preview_mesh_vtk_batch(self, trimesh, mode, index):
        """
        Export mesh from batch and prepare for VTK.js preview.

        Supports two visualization modes:
        - fields: Scientific visualization with scalar fields and colormaps (VTP/STL export)
        - texture: Textured mesh visualization with materials (GLB export)

        Args:
            trimesh: List of trimesh_module.Trimesh objects
            mode: List with visualization mode - "fields" or "texture"
            index: List with current index to display

        Returns:
            dict: UI data for frontend widget
        """
        # Extract values from lists (ComfyUI passes inputs as lists when INPUT_IS_LIST=True)
        mode_val = mode[0] if isinstance(mode, list) else mode
        index_val = index[0] if isinstance(index, list) else index

        # Validate batch
        if not trimesh or len(trimesh) == 0:
            raise ValueError("Empty mesh batch provided")

        # Clamp index to valid range
        batch_size = len(trimesh)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current mesh from batch
        current_mesh = trimesh[actual_index]

        print(f"[PreviewMeshVTKBatch] Batch size: {batch_size}, showing index: {actual_index + 1}/{batch_size}")
        print(f"[PreviewMeshVTKBatch] Preparing preview: {get_geometry_type(current_mesh)} - {len(current_mesh.vertices)} vertices, {get_face_count(current_mesh)} faces")

        # Check for scalar fields (vertex/face attributes)
        has_vertex_attrs = hasattr(current_mesh, 'vertex_attributes') and len(current_mesh.vertex_attributes) > 0
        has_face_attrs = hasattr(current_mesh, 'face_attributes') and len(current_mesh.face_attributes) > 0
        has_fields = has_vertex_attrs or has_face_attrs

        print(f"[PreviewMeshVTKBatch] DEBUG - hasattr vertex_attributes: {hasattr(current_mesh, 'vertex_attributes')}")
        print(f"[PreviewMeshVTKBatch] DEBUG - hasattr face_attributes: {hasattr(current_mesh, 'face_attributes')}")
        if hasattr(current_mesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKBatch] DEBUG - vertex_attributes: {current_mesh.vertex_attributes}")
            print(f"[PreviewMeshVTKBatch] DEBUG - len(vertex_attributes): {len(current_mesh.vertex_attributes)}")
        if hasattr(current_mesh, 'face_attributes'):
            print(f"[PreviewMeshVTKBatch] DEBUG - face_attributes: {current_mesh.face_attributes}")
            print(f"[PreviewMeshVTKBatch] DEBUG - len(face_attributes): {len(current_mesh.face_attributes)}")
        print(f"[PreviewMeshVTKBatch] DEBUG - has_vertex_attrs: {has_vertex_attrs}")
        print(f"[PreviewMeshVTKBatch] DEBUG - has_face_attrs: {has_face_attrs}")
        print(f"[PreviewMeshVTKBatch] DEBUG - has_fields: {has_fields}")

        # Check for visual data (textures/vertex colors)
        has_visual = hasattr(current_mesh, 'visual') and current_mesh.visual is not None
        visual_kind = current_mesh.visual.kind if has_visual else None
        has_texture = visual_kind == 'texture' and hasattr(current_mesh.visual, 'material') if has_visual else False
        has_vertex_colors = visual_kind == 'vertex' if has_visual else False
        has_material = has_texture

        print(f"[PreviewMeshVTKBatch] Mode: {mode_val}")
        print(f"[PreviewMeshVTKBatch] Visual data - has_visual: {has_visual}, kind: {visual_kind}, texture: {has_texture}, vertex_colors: {has_vertex_colors}")

        # Check if this is a point cloud
        is_pc = is_point_cloud(current_mesh)

        # Choose export format based on visualization mode
        if mode_val == "texture":
            # Texture mode: Export GLB to preserve textures/materials/UVs
            filename = f"preview_vtk_batch_{uuid.uuid4().hex[:8]}.glb"
            viewer_type = "texture"
            print(f"[PreviewMeshVTKBatch] Using texture mode - GLB export")
        else:
            # Fields mode: Export VTP/STL for scalar field visualization
            if has_fields or is_pc:
                # Export to VTP for: scalar fields OR point clouds (STL doesn't support point clouds)
                filename = f"preview_vtk_batch_{uuid.uuid4().hex[:8]}.vtp"
                print(f"[PreviewMeshVTKBatch] Using VTP format (fields={has_fields}, point_cloud={is_pc})")
            else:
                # Export to STL (compact format for simple surface meshes)
                filename = f"preview_vtk_batch_{uuid.uuid4().hex[:8]}.stl"
            viewer_type = "fields"
            print(f"[PreviewMeshVTKBatch] Using fields mode - VTP/STL export")

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh
        try:
            if mode_val == "texture":
                # Export GLB for texture rendering
                current_mesh.export(filepath, file_type='glb', include_normals=True)
                print(f"[PreviewMeshVTKBatch] Exported GLB to: {filepath}")
            elif has_fields or is_pc:
                # Use VTP exporter for fields or point clouds
                export_mesh_with_scalars_vtp(current_mesh, filepath)
                print(f"[PreviewMeshVTKBatch] Exported VTP to: {filepath}")
            else:
                # Use STL for simple surface meshes
                current_mesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshVTKBatch] Exported STL to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKBatch] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            current_mesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTKBatch] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = current_mesh.bounds
        extents = current_mesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight (only for actual meshes, not point clouds)
        is_watertight = False if is_point_cloud(current_mesh) else current_mesh.is_watertight

        # Calculate volume and area (only for meshes with faces, not point clouds)
        volume = None
        area = None
        if not is_point_cloud(current_mesh):
            try:
                if is_watertight:
                    volume = float(current_mesh.volume)
                area = float(current_mesh.area)
            except Exception as e:
                print(f"[PreviewMeshVTKBatch] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays) - for field visualization UI
        field_names = []
        if has_vertex_attrs:
            field_names.extend(list(current_mesh.vertex_attributes.keys()))
            print(f"[PreviewMeshVTKBatch] Vertex attributes: {list(current_mesh.vertex_attributes.keys())}")
        if has_face_attrs:
            field_names.extend([f"face.{k}" for k in current_mesh.face_attributes.keys()])
            print(f"[PreviewMeshVTKBatch] Face attributes: {list(current_mesh.face_attributes.keys())}")

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "viewer_type": [viewer_type],  # "fields" or "texture" - tells frontend which viewer to load
            "mode": [mode_val],  # User-selected mode
            "vertex_count": [len(current_mesh.vertices)],
            "face_count": [get_face_count(current_mesh)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
            # Batch-specific metadata
            "batch_size": [batch_size],
            "current_index": [actual_index],
        }

        # Add mode-specific metadata
        if viewer_type == "texture":
            # Texture mode metadata
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

        if viewer_type == "texture":
            print(f"[PreviewMeshVTKBatch] Texture mode info: watertight={is_watertight}, volume={volume}, area={area}, texture={has_texture}, vertex_colors={has_vertex_colors}")
        elif field_names:
            print(f"[PreviewMeshVTKBatch] Fields mode info: watertight={is_watertight}, volume={volume}, area={area}, fields={field_names}")
        else:
            print(f"[PreviewMeshVTKBatch] Fields mode info: watertight={is_watertight}, volume={volume}, area={area}, no fields")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTKBatch": PreviewMeshVTKBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTKBatch": "Preview Mesh Batch (VTK)",
}
