# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Save Mesh Node - Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
"""

import logging
import os

log = logging.getLogger("geometrypack")

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_OUTPUT_FOLDER = None

from . import mesh_io
from comfy_api.latest import io


class SaveMesh(io.ComfyNode):
    """
    Save a mesh or point cloud to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    Point clouds (vertices without faces) can be saved as PLY format.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSaveMesh",
            display_name="Save Mesh",
            category="geompack/io",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("file_path", default="output", multiline=False, tooltip="Output filename (without extension) or path"),
                io.Combo.Input("format", options=["obj", "ply", "stl", "off", "glb", "gltf", "vtp"], default="obj", tooltip="Output file format"),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, file_path, format="obj"):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            trimesh: trimesh.Trimesh object
            file_path: Output file path (relative to output folder or absolute)
            format: Output format (obj, ply, stl, off, glb, gltf, vtp)

        Returns:
            tuple: (status_message,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Ensure file has correct extension
        expected_ext = f".{format}"
        if not file_path.lower().endswith(expected_ext):
            # Remove any existing extension and add the correct one
            base_path = os.path.splitext(file_path)[0]
            file_path = base_path + expected_ext

        # Debug: Check what we received
        log.debug("Received mesh type: %s", type(trimesh))
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None instead of a mesh object. Check that the previous node is outputting a mesh.")

        # Check if mesh has data
        is_point_cloud = False
        try:
            vertex_count = len(trimesh.vertices) if hasattr(trimesh, 'vertices') else 0
            face_count = len(trimesh.faces) if hasattr(trimesh, 'faces') else 0
            log.info("Mesh has %d vertices, %d faces", vertex_count, face_count)

            if vertex_count == 0:
                raise ValueError(
                    f"Cannot save empty geometry (vertices: {vertex_count}). "
                    "Check that the previous node is producing valid geometry."
                )

            # Point cloud (no faces) - only PLY format supports this well
            is_point_cloud = face_count == 0
            if is_point_cloud and format not in ["ply"]:
                log.warning("Point cloud detected but format is '%s'. "
                            "Switching to PLY format for point cloud export.", format)
                format = "ply"
                # Update file path extension
                base_path = os.path.splitext(file_path)[0]
                file_path = base_path + ".ply"

        except Exception as e:
            raise ValueError(f"Error checking mesh properties: {e}. Received object may not be a valid mesh.")

        # Determine full output path
        full_path = file_path

        # If path is relative and we have output folder, use it
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            log.info("Saving to output folder: %s", file_path)
        else:
            log.info("Saving to: %s", file_path)

        # Save the mesh
        success, error = mesh_io.save_mesh_file(trimesh, full_path)

        if not success:
            raise ValueError(f"Failed to save trimesh: {error}")

        geom_type = "point cloud" if is_point_cloud else "mesh"
        log.info("Successfully saved %s to: %s", geom_type, full_path)
        log.info("  Vertices: %d", len(trimesh.vertices))
        if not is_point_cloud:
            log.info("  Faces: %d", len(trimesh.faces))

        return io.NodeOutput(full_path)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackSaveMesh": SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSaveMesh": "Save Mesh",
}
