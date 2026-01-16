# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh Blend Node - Load Blender .blend files using direct bpy via comfy-env isolation.
"""

import os
import trimesh as trimesh_module

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None


class LoadMeshBlend:
    """
    Load Blender .blend files using direct bpy via comfy-env isolation.

    Uses the bpy Python module in an isolated environment to directly
    extract mesh data from .blend files without subprocess or temp files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of .blend files only
        blend_files = cls.get_blend_files()

        if not blend_files:
            blend_files = ["No .blend files found in input/3d or input folders"]

        return {
            "required": {
                "file_path": (blend_files, ),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_blend"
    CATEGORY = "geompack/io"

    @classmethod
    def get_blend_files(cls):
        """Get list of available .blend files in input/3d and input folders."""
        blend_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if file.lower().endswith('.blend'):
                        blend_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path) and file.lower().endswith('.blend'):
                    blend_files.append(file)

        return sorted(blend_files)

    @classmethod
    def IS_CHANGED(cls, file_path):
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

    def load_blend(self, file_path):
        """
        Load .blend file using direct bpy via comfy-env isolation.

        Args:
            file_path: Path to .blend file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh, info_string)
        """
        from .._utils.bpy_worker import call_bpy

        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the .blend file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[LoadMeshBlend] Found .blend in input/3d folder: {file_path}")

            # Second, try in ComfyUI input folder
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[LoadMeshBlend] Found .blend in input folder: {file_path}")

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[LoadMeshBlend] Loading from absolute path: {file_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load .blend file using bpy_worker
        print(f"[LoadMeshBlend] Loading via bpy isolated: {full_path}")
        try:
            result = call_bpy('bpy_import_blend', blend_path=full_path)
        except Exception as e:
            raise ValueError(f"Failed to load .blend file: {e}")

        if len(result['vertices']) == 0:
            raise ValueError(f"No mesh data found in .blend file: {full_path}")

        import numpy as np
        loaded_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        # Add metadata
        loaded_mesh.metadata['source'] = {
            'file': os.path.basename(full_path),
            'format': 'blend',
            'loader': 'bpy_isolated'
        }

        # Generate info string
        info = f"Blender File Loaded (bpy isolated)\n"
        info += f"File: {os.path.basename(full_path)}\n"
        info += f"Vertices: {len(loaded_mesh.vertices):,}\n"
        info += f"Faces: {len(loaded_mesh.faces):,}"

        print(f"[LoadMeshBlend] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshBlend": LoadMeshBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshBlend": "Load Mesh (Blender)",
}
