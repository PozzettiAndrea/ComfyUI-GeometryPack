# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh FBX Node - Load FBX files using bpy.
"""

import logging
import os

# Import bpy first so its bundled tbb12.dll wins the loader race against
# trimesh[easy]'s embreex.libs/tbb12.dll. Reverse order produces
# STATUS_ENTRYPOINT_NOT_FOUND when bpy's C extension calls into the wrong
# tbb build.
import bpy  # noqa: F401  (loaded for DLL-ordering side-effect)

import numpy as np
import trimesh as trimesh_module

log = logging.getLogger("geometrypack")

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
from comfy_api.latest import io


def _bpy_import_fbx(fbx_path):
    """Import .fbx file and extract mesh data using bpy."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        return {'vertices': [], 'faces': [], 'name': 'empty'}

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in mesh_objects:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj.matrix_world)

        verts = [list(v.co) for v in mesh.vertices]

        for poly in mesh.polygons:
            if len(poly.vertices) == 3:
                all_faces.append([
                    poly.vertices[0] + vertex_offset,
                    poly.vertices[1] + vertex_offset,
                    poly.vertices[2] + vertex_offset
                ])
            elif len(poly.vertices) > 3:
                v0 = poly.vertices[0]
                for i in range(1, len(poly.vertices) - 1):
                    all_faces.append([
                        v0 + vertex_offset,
                        poly.vertices[i] + vertex_offset,
                        poly.vertices[i+1] + vertex_offset
                    ])

        all_vertices.extend(verts)
        vertex_offset += len(verts)
        obj_eval.to_mesh_clear()

    return {
        'vertices': all_vertices,
        'faces': all_faces,
        'name': mesh_objects[0].name if mesh_objects else 'combined'
    }


class LoadMeshFBX(io.ComfyNode):
    """
    Load FBX files using bpy.

    Uses the bpy Python module to directly import FBX files and extract mesh data.
    """


    @classmethod
    def define_schema(cls):
        fbx_files = cls.get_fbx_files()
        if not fbx_files:
            fbx_files = ["No FBX files found in input/3d or input folders"]
        return io.Schema(
            node_id="GeomPackLoadMeshFBX",
            display_name="Load Mesh (FBX)",
            category="geompack/io",
            is_output_node=True,
            inputs=[
                io.Combo.Input("file_path", options=fbx_files),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def get_fbx_files(cls):
        """Get list of available FBX files in input/3d and input folders."""
        fbx_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if file.lower().endswith('.fbx'):
                        fbx_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path) and file.lower().endswith('.fbx'):
                    fbx_files.append(file)

        return sorted(fbx_files)

    @classmethod
    def fingerprint_inputs(cls, file_path):
        """Force re-execution when file changes."""
        if COMFYUI_INPUT_FOLDER is not None:
            # Check file modification time
            full_path = None
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)

            if os.path.exists(input_3d_path):
                full_path = input_3d_path
            elif os.path.exists(input_path):
                full_path = input_path

            if full_path and os.path.exists(full_path):
                return os.path.getmtime(full_path)

        return file_path

    @classmethod
    def execute(cls, file_path):
        """
        Load FBX file using bpy.

        Args:
            file_path: Path to FBX file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh, info_string)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the FBX file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                log.info("Found FBX in input/3d folder: %s", file_path)

            # Second, try in ComfyUI input folder
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    log.info("Found FBX in input folder: %s", file_path)

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                log.info("Loading from absolute path: %s", file_path)
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load FBX file using bpy directly
        log.info("Loading via bpy: %s", full_path)
        try:
            result = _bpy_import_fbx(full_path)
        except Exception as e:
            raise ValueError(f"Failed to load FBX file: {e}")

        if len(result['vertices']) == 0:
            raise ValueError(f"No mesh data found in FBX file: {full_path}")

        loaded_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        # Add metadata
        loaded_mesh.metadata['source'] = {
            'file': os.path.basename(full_path),
            'format': 'fbx',
            'loader': 'bpy'
        }

        # Generate info string
        info = f"FBX Loaded (bpy)\n"
        info += f"File: {os.path.basename(full_path)}\n"
        info += f"Vertices: {len(loaded_mesh.vertices):,}\n"
        info += f"Faces: {len(loaded_mesh.faces):,}"

        log.info("Loaded: %d vertices, %d faces", len(loaded_mesh.vertices), len(loaded_mesh.faces))

        return io.NodeOutput(loaded_mesh, info, ui={"text": [info]})


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshFBX": LoadMeshFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshFBX": "Load Mesh (FBX)",
}
