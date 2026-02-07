# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Save Mesh Batch Node - Save multiple meshes to a folder with sequential numbering.
"""

import os
import math

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_OUTPUT_FOLDER = None

from . import mesh_io


class SaveMeshBatch:
    """
    Save multiple meshes to a folder with sequential numbering.

    Files are named: {base_name}_{number}.{format}
    Number of digits is automatically determined based on batch size:
    - 1-9 meshes: _1, _2, ...
    - 10-99 meshes: _01, _02, ...
    - 100-999 meshes: _001, _002, ...
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "folder_name": ("STRING", {
                    "default": "mesh_output",
                    "multiline": False,
                    "tooltip": "Name of the folder to create in the output directory"
                }),
                "base_name": ("STRING", {
                    "default": "mesh",
                    "multiline": False,
                    "tooltip": "Base name for files (e.g., 'mesh' becomes 'mesh_001.obj'). Ignored if names provided."
                }),
                "format": (["obj", "ply", "stl", "off", "glb", "gltf", "vtp"],),
            },
            "optional": {
                "names": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional list of custom filenames (without extension). If provided, overrides base_name."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_folder", "saved_count")
    FUNCTION = "save_mesh_batch"
    CATEGORY = "geompack/io"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    def save_mesh_batch(self, trimesh, folder_name, base_name, format, names=None):
        """
        Save batch of meshes to folder with sequential numbering or custom names.

        Args:
            trimesh: List of trimesh.Trimesh objects
            folder_name: Name of folder to create
            base_name: Base name for files (used if names not provided)
            format: Output format (obj, ply, stl, etc.)
            names: Optional list of custom filenames (without extension)

        Returns:
            tuple: (output_folder_path, saved_count)
        """
        import re

        # Extract values from lists (ComfyUI passes inputs as lists when INPUT_IS_LIST=True)
        folder_name_val = folder_name[0] if isinstance(folder_name, list) else folder_name
        base_name_val = base_name[0] if isinstance(base_name, list) else base_name
        format_val = format[0] if isinstance(format, list) else format

        # Handle names list - it comes as a list when INPUT_IS_LIST=True
        names_list = None
        if names is not None and len(names) > 0:
            names_list = names  # Already a list due to INPUT_IS_LIST=True

        # Validate inputs
        if not trimesh or len(trimesh) == 0:
            raise ValueError("No meshes provided to save")

        if not folder_name_val or folder_name_val.strip() == "":
            raise ValueError("Folder name cannot be empty")

        if not base_name_val or base_name_val.strip() == "":
            raise ValueError("Base name cannot be empty")

        batch_size = len(trimesh)
        use_custom_names = names_list is not None and len(names_list) >= batch_size
        if use_custom_names:
            print(f"[SaveMeshBatch] Saving {batch_size} meshes with custom names to folder '{folder_name_val}'")
        else:
            print(f"[SaveMeshBatch] Saving {batch_size} meshes to folder '{folder_name_val}'")

        # Determine output folder path
        if COMFYUI_OUTPUT_FOLDER is not None:
            output_folder = os.path.join(COMFYUI_OUTPUT_FOLDER, folder_name_val)
        else:
            output_folder = folder_name_val

        # Create folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"[SaveMeshBatch] Output folder: {output_folder}")

        # Calculate number of digits needed based on batch size
        # For 1-9: 1 digit, 10-99: 2 digits, 100-999: 3 digits, etc.
        num_digits = max(1, math.ceil(math.log10(batch_size + 1)))
        print(f"[SaveMeshBatch] Using {num_digits} digits for numbering (batch size: {batch_size})")

        saved_count = 0
        errors = []

        for i, mesh in enumerate(trimesh):
            # Generate filename - use custom name if available, otherwise sequential
            if use_custom_names and i < len(names_list):
                # Sanitize custom name (remove invalid filename characters)
                safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(names_list[i]))
                filename = f"{safe_name}.{format_val}"
            else:
                file_number = str(i + 1).zfill(num_digits)
                filename = f"{base_name_val}_{file_number}.{format_val}"
            file_path = os.path.join(output_folder, filename)

            # Validate mesh
            if mesh is None:
                errors.append(f"Mesh {i + 1}: None object")
                continue

            try:
                vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
                face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0

                if vertex_count == 0 or face_count == 0:
                    errors.append(f"Mesh {i + 1}: Empty mesh (vertices: {vertex_count}, faces: {face_count})")
                    continue

                # Save the mesh
                success, error = mesh_io.save_mesh_file(mesh, file_path)

                if success:
                    saved_count += 1
                    if (i + 1) % 10 == 0 or i == 0 or i == batch_size - 1:
                        print(f"[SaveMeshBatch] Saved {filename} ({vertex_count} verts, {face_count} faces)")
                else:
                    errors.append(f"Mesh {i + 1}: {error}")

            except Exception as e:
                errors.append(f"Mesh {i + 1}: {str(e)}")

        # Report results
        print(f"[SaveMeshBatch] Saved {saved_count}/{batch_size} meshes to {output_folder}")

        if errors:
            print(f"[SaveMeshBatch] Errors ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

        if saved_count == 0:
            raise ValueError(f"Failed to save any meshes. Errors: {'; '.join(errors[:3])}")

        return (output_folder, saved_count)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackSaveMeshBatch": SaveMeshBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSaveMeshBatch": "Save Meshes to Folder",
}
