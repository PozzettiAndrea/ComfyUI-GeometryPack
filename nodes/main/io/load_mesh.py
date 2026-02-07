# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh Node - Load a mesh from file (OBJ, PLY, STL, OFF, etc.)
"""

import os
import numpy as np

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None

from . import mesh_io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LoadMesh:
    """
    Load a mesh from file (OBJ, PLY, STL, OFF, etc.)
    Now returns trimesh.Trimesh objects for better mesh handling.
    """

    # Supported mesh extensions for file browser
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available mesh files (like LoadImage does)
        mesh_files = cls.get_mesh_files()

        # If no files found, provide a default empty list
        if not mesh_files:
            mesh_files = ["No mesh files found in input/3d or input folders"]

        return {
            "required": {
                "file_path": (mesh_files, ),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh", "texture")
    FUNCTION = "load_mesh"
    CATEGORY = "geompack/io"

    @classmethod
    def get_mesh_files(cls):
        """Get list of available mesh files in input/3d and input folders."""
        mesh_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        return sorted(mesh_files)

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

    def _extract_texture_image(self, mesh):
        """Extract texture from mesh and convert to ComfyUI IMAGE format."""
        if not PIL_AVAILABLE:
            return None

        texture_image = None

        # Check if mesh has texture
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if material is not None:
                # Check for PBR baseColorTexture (GLB/GLTF files)
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    img = material.baseColorTexture
                    if isinstance(img, Image.Image):
                        texture_image = img
                        print(f"[LoadMesh] Found texture in material.baseColorTexture: {texture_image.size}")
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)
                        print(f"[LoadMesh] Loaded texture from material.baseColorTexture path: {texture_image.size}")

                # Check for standard material.image (OBJ/MTL files)
                if texture_image is None and hasattr(material, 'image') and material.image is not None:
                    img = material.image
                    if isinstance(img, Image.Image):
                        texture_image = img
                        print(f"[LoadMesh] Found texture in material.image: {texture_image.size}")
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)
                        print(f"[LoadMesh] Loaded texture from material.image path: {texture_image.size}")

        if texture_image is None:
            print("[LoadMesh] No texture found in mesh")
            # Return black 64x64 placeholder
            texture_image = Image.new('RGB', (64, 64), color=(0, 0, 0))

        # Convert to ComfyUI IMAGE format (BHWC with values 0-1)
        img_array = np.array(texture_image.convert("RGB")).astype(np.float32) / 255.0
        return img_array[np.newaxis, ...]

    def load_mesh(self, file_path):
        """
        Load mesh from file.

        Looks for files in ComfyUI's input/3d folder first, then input folder, then tries absolute path.

        Args:
            file_path: Path to mesh file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh, IMAGE)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[LoadMesh] Found mesh in input/3d folder: {file_path}")

            # Second, try in ComfyUI input folder (for backward compatibility)
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[LoadMesh] Found mesh in input folder: {file_path}")

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[LoadMesh] Loading from absolute path: {file_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load the mesh
        loaded_mesh, error = mesh_io.load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        # Handle both meshes and pointclouds
        if hasattr(loaded_mesh, 'faces') and loaded_mesh.faces is not None:
            print(f"[LoadMesh] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")
        else:
            print(f"[LoadMesh] Loaded pointcloud: {len(loaded_mesh.vertices)} points")

        # Extract texture
        texture = self._extract_texture_image(loaded_mesh)

        return (loaded_mesh, texture)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMesh": LoadMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMesh": "Load Mesh",
}
