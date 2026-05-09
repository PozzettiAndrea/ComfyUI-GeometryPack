# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh (Path) Node - Load a mesh from a string path input
"""

import logging
import os
import numpy as np

log = logging.getLogger("geometrypack")

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

from . import mesh_io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
from comfy_api.latest import io


class LoadMeshPath(io.ComfyNode):
    """
    Load a mesh from a string path (OBJ, PLY, STL, OFF, etc.)
    Takes a string input for the path, allowing dynamic path construction.

    Supports batch paths: pass multiple paths separated by newlines or commas.
    When multiple paths are provided, returns lists of meshes and textures.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackLoadMeshPath",
            display_name="Load Mesh (Path)",
            category="geompack/io",
            description='Load mesh(es) from path(s). Supports batch paths (newline or comma separated).',
            inputs=[
                io.String.Input("file_path", default="", multiline=True, tooltip="Path to mesh file(s). Supports multiple paths separated by newlines or commas."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh", is_output_list=True),
                io.Image.Output(display_name="texture", is_output_list=True),
            ],
        )

    @classmethod
    def _parse_paths(cls, file_path_input):
        """Parse input into list of paths, handling newlines and commas."""
        if not file_path_input or file_path_input.strip() == "":
            return []

        # Split by newlines first, then by commas if no newlines
        if '\n' in file_path_input:
            paths = file_path_input.strip().split('\n')
        elif ',' in file_path_input:
            paths = file_path_input.strip().split(',')
        else:
            paths = [file_path_input]

        # Clean up paths and filter empty ones
        paths = [p.strip() for p in paths if p.strip()]
        return paths

    @classmethod
    def fingerprint_inputs(cls, file_path):
        """Force re-execution when any file changes."""
        paths = cls._parse_paths(file_path)
        mtimes = []
        for path in paths:
            resolved_path = cls._resolve_path(path)
            if resolved_path and os.path.exists(resolved_path):
                mtimes.append(str(os.path.getmtime(resolved_path)))
            else:
                mtimes.append(path)
        return "_".join(mtimes)

    @classmethod
    def _resolve_path(cls, file_path):
        """Resolve file path, checking multiple locations."""
        if not file_path or file_path.strip() == "":
            return None

        file_path = file_path.strip()

        # Try absolute path first
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path

        # Try relative to output folder (common for generated meshes)
        if COMFYUI_OUTPUT_FOLDER is not None:
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            if os.path.exists(output_path):
                return output_path

        # Try relative to input/3d folder
        if COMFYUI_INPUT_FOLDER is not None:
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            if os.path.exists(input_3d_path):
                return input_3d_path

            # Try relative to input folder
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
            if os.path.exists(input_path):
                return input_path

        # Try as-is (might be absolute path that exists)
        if os.path.exists(file_path):
            return file_path

        return None

    @staticmethod
    def _extract_texture_image(mesh):
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
                        log.debug("Found texture in material.baseColorTexture: %s", texture_image.size)
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)
                        log.debug("Loaded texture from material.baseColorTexture path: %s", texture_image.size)

                # Check for standard material.image (OBJ/MTL files)
                if texture_image is None and hasattr(material, 'image') and material.image is not None:
                    img = material.image
                    if isinstance(img, Image.Image):
                        texture_image = img
                        log.debug("Found texture in material.image: %s", texture_image.size)
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)
                        log.debug("Loaded texture from material.image path: %s", texture_image.size)

        if texture_image is None:
            log.debug("No texture found in mesh")
            # Return black 64x64 placeholder
            texture_image = Image.new('RGB', (64, 64), color=(0, 0, 0))

        # Convert to ComfyUI IMAGE format (BHWC with values 0-1)
        img_array = np.array(texture_image.convert("RGB")).astype(np.float32) / 255.0
        return img_array[np.newaxis, ...]

    @staticmethod
    def _load_single_mesh(file_path):
        """Load a single mesh from file path string."""
        file_path = file_path.strip()

        # Resolve the path
        full_path = LoadMeshPath._resolve_path(file_path)

        if full_path is None:
            # Build error message with searched paths
            searched_paths = [file_path]
            if COMFYUI_OUTPUT_FOLDER:
                searched_paths.append(os.path.join(COMFYUI_OUTPUT_FOLDER, file_path))
            if COMFYUI_INPUT_FOLDER:
                searched_paths.append(os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path))
                searched_paths.append(os.path.join(COMFYUI_INPUT_FOLDER, file_path))

            error_msg = f"File not found: '{file_path}'\nSearched in:"
            for path in searched_paths:
                error_msg += f"\n  - {path}"
            raise ValueError(error_msg)

        log.info("Loading mesh from: %s", full_path)

        # Load the mesh
        loaded_mesh, error = mesh_io.load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        # Handle both meshes and pointclouds
        if hasattr(loaded_mesh, 'faces') and loaded_mesh.faces is not None:
            log.info("Loaded: %d vertices, %d faces", len(loaded_mesh.vertices), len(loaded_mesh.faces))
        else:
            log.info("Loaded pointcloud: %d points", len(loaded_mesh.vertices))

        # Extract texture
        texture = LoadMeshPath._extract_texture_image(loaded_mesh)

        return (loaded_mesh, texture)

    @classmethod
    def execute(cls, file_path):
        """
        Load mesh(es) from file path string(s).

        Args:
            file_path: Path to mesh file(s). Can be single path or multiple paths
                       separated by newlines or commas.

        Returns:
            tuple: (list of trimesh.Trimesh, list of IMAGE)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Parse paths
        paths = cls._parse_paths(file_path)

        if not paths:
            raise ValueError("No valid paths provided")

        log.info("Loading %d mesh(es)", len(paths))

        meshes = []
        textures = []

        for i, path in enumerate(paths):
            try:
                mesh, texture = cls._load_single_mesh(path)
                meshes.append(mesh)
                textures.append(texture)
            except Exception as e:
                log.error("Error loading mesh %d (%s): %s", i + 1, path, e)
                # Continue with other paths instead of failing completely
                raise

        log.info("Successfully loaded %d mesh(es)", len(meshes))
        return io.NodeOutput(meshes, textures)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshPath": LoadMeshPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshPath": "Load Mesh (Path)",
}
