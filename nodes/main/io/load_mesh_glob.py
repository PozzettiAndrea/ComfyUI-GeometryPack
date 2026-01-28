# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh (Glob) Node - Load meshes matching a glob pattern
"""

import os
import glob as glob_module
import numpy as np

from ..._utils import mesh_ops

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _get_torch():
    """Lazy torch import to avoid importing before ComfyUI startup."""
    try:
        import torch
        return torch
    except ImportError:
        return None


class LoadMeshGlob:
    """
    Load meshes matching a glob pattern (e.g., /path/*.glb, /path/**/*.obj)
    Returns a list of meshes sorted by filename.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glob_pattern": ("STRING", {
                    "default": "",
                    "tooltip": "Glob pattern to match mesh files (e.g., /path/to/folder/*.glb)"
                }),
            },
            "optional": {
                "sort_by": (["name", "modified_time"], {
                    "default": "name",
                    "tooltip": "How to sort matched files"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "STRING")
    RETURN_NAMES = ("meshes", "textures", "file_paths")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load_meshes"
    CATEGORY = "geompack/io"
    DESCRIPTION = "Load all meshes matching a glob pattern"

    @classmethod
    def IS_CHANGED(cls, glob_pattern, sort_by="name"):
        """Force re-execution when any matched file changes."""
        matched_files = glob_module.glob(glob_pattern, recursive=True)
        mtimes = []
        for path in matched_files:
            if os.path.exists(path):
                mtimes.append(f"{path}:{os.path.getmtime(path)}")
        return "_".join(sorted(mtimes))

    def _extract_texture_image(self, mesh):
        """Extract texture from mesh and convert to ComfyUI IMAGE format."""
        torch = _get_torch()
        if not PIL_AVAILABLE or torch is None:
            return self._placeholder_texture()

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
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)

                # Check for standard material.image (OBJ/MTL files)
                if texture_image is None and hasattr(material, 'image') and material.image is not None:
                    img = material.image
                    if isinstance(img, Image.Image):
                        texture_image = img
                    elif isinstance(img, str) and os.path.exists(img):
                        texture_image = Image.open(img)

        if texture_image is None:
            return self._placeholder_texture()

        # Convert to ComfyUI IMAGE format (BHWC with values 0-1)
        img_array = np.array(texture_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    def _placeholder_texture(self):
        """Return a black 64x64 placeholder texture."""
        torch = _get_torch()
        if torch is None:
            return None
        return torch.zeros(1, 64, 64, 3)

    def load_meshes(self, glob_pattern, sort_by="name"):
        """
        Load meshes matching the glob pattern.

        Args:
            glob_pattern: Glob pattern to match mesh files
            sort_by: How to sort matched files ("name" or "modified_time")

        Returns:
            tuple: (list of trimesh.Trimesh, list of IMAGE, list of file paths)
        """
        if not glob_pattern or glob_pattern.strip() == "":
            raise ValueError("Glob pattern cannot be empty")

        glob_pattern = glob_pattern.strip()

        # Find matching files
        matched_files = glob_module.glob(glob_pattern, recursive=True)

        if not matched_files:
            print(f"[LoadMeshGlob] No files matched pattern: {glob_pattern}")
            return ([], [], [])

        # Sort files
        if sort_by == "name":
            matched_files.sort()
        else:
            matched_files.sort(key=os.path.getmtime)

        print(f"[LoadMeshGlob] Found {len(matched_files)} files matching pattern")

        meshes = []
        textures = []
        file_paths = []

        for path in matched_files:
            try:
                print(f"[LoadMeshGlob] Loading: {path}")
                mesh, error = mesh_ops.load_mesh_file(path)

                if mesh is None:
                    print(f"[LoadMeshGlob] Failed to load {path}: {error}")
                    continue

                # Handle both meshes and pointclouds
                if hasattr(mesh, 'faces') and mesh.faces is not None:
                    print(f"[LoadMeshGlob] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                else:
                    print(f"[LoadMeshGlob] Loaded pointcloud: {len(mesh.vertices)} points")

                meshes.append(mesh)
                textures.append(self._extract_texture_image(mesh))
                file_paths.append(path)

            except Exception as e:
                print(f"[LoadMeshGlob] Error loading {path}: {e}")
                continue

        print(f"[LoadMeshGlob] Successfully loaded {len(meshes)} mesh(es)")
        return (meshes, textures, file_paths)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshGlob": LoadMeshGlob,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshGlob": "Load Mesh Batch (Glob)",
}
