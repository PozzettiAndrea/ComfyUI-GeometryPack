# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh (Glob) Node - Load meshes matching a glob pattern
"""

import logging
import os
import glob as glob_module
import numpy as np

log = logging.getLogger("geometrypack")

from . import mesh_io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
from comfy_api.latest import io


class LoadMeshGlob(io.ComfyNode):
    """
    Load meshes matching a glob pattern (e.g., /path/*.glb, /path/**/*.obj)
    Returns a list of meshes sorted by filename.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackLoadMeshGlob",
            display_name="Load Mesh Batch (Glob)",
            category="geompack/io",
            description='Load all meshes matching a glob pattern',
            inputs=[
                io.String.Input("glob_pattern", default="", tooltip="Glob pattern to match mesh files (e.g., /path/to/folder/*.glb)"),
                io.Combo.Input("sort_by", options=["name", "modified_time"], default="name", tooltip="How to sort matched files", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
                io.Image.Output(display_name="textures", is_output_list=True),
                io.String.Output(display_name="file_paths", is_output_list=True),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, glob_pattern, sort_by="name"):
        """Force re-execution when any matched file changes."""
        matched_files = glob_module.glob(glob_pattern, recursive=True)
        mtimes = []
        for path in matched_files:
            if os.path.exists(path):
                mtimes.append(f"{path}:{os.path.getmtime(path)}")
        return "_".join(sorted(mtimes))

    @staticmethod
    def _extract_texture_image(mesh):
        """Extract texture from mesh and convert to ComfyUI IMAGE format."""
        if not PIL_AVAILABLE:
            return LoadMeshGlob._placeholder_texture()

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
            return LoadMeshGlob._placeholder_texture()

        # Convert to ComfyUI IMAGE format (BHWC with values 0-1)
        img_array = np.array(texture_image.convert("RGB")).astype(np.float32) / 255.0
        return img_array[np.newaxis, ...]

    @staticmethod
    def _placeholder_texture():
        """Return a black 64x64 placeholder texture."""
        return np.zeros((1, 64, 64, 3), dtype=np.float32)

    @classmethod
    def execute(cls, glob_pattern, sort_by="name"):
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
            log.warning("No files matched pattern: %s", glob_pattern)
            return io.NodeOutput([], [], [])

        # Sort files
        if sort_by == "name":
            matched_files.sort()
        else:
            matched_files.sort(key=os.path.getmtime)

        log.info("Found %d files matching pattern", len(matched_files))

        meshes = []
        textures = []
        file_paths = []

        for path in matched_files:
            try:
                log.info("Loading: %s", path)
                mesh, error = mesh_io.load_mesh_file(path)

                if mesh is None:
                    log.warning("Failed to load %s: %s", path, error)
                    continue

                # Handle both meshes and pointclouds
                if hasattr(mesh, 'faces') and mesh.faces is not None:
                    log.info("Loaded: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
                else:
                    log.info("Loaded pointcloud: %d points", len(mesh.vertices))

                meshes.append(mesh)
                textures.append(cls._extract_texture_image(mesh))
                file_paths.append(path)

            except Exception as e:
                log.error("Error loading %s: %s", path, e)
                continue

        log.info("Successfully loaded %d mesh(es)", len(meshes))
        return io.NodeOutput(meshes, textures, file_paths)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshGlob": LoadMeshGlob,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshGlob": "Load Mesh Batch (Glob)",
}
