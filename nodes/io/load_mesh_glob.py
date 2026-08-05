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
                io.String.Input("glob_pattern", default="", tooltip="Glob pattern to match mesh files (e.g., /path/to/folder/*.glb). Relative patterns resolve from ComfyUI's input directory; absolute paths are used as-is."),
                io.Combo.Input("sort_by", options=["name", "modified_time"], default="name", tooltip="How to sort matched files", optional=True),
                io.Int.Input("start_index", default=0, min=0, max=100000, tooltip="Skip the first N matched files"),
                io.Int.Input("max_meshes", default=-1, min=-1, max=100000, tooltip="Load up to N files after start_index (-1 = unlimited)"),
                io.Int.Input("num_workers", default=1, min=1, max=16, tooltip="Number of files to load concurrently (1 = sequential, same as before)"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
                io.Image.Output(display_name="textures", is_output_list=True),
                io.String.Output(display_name="file_paths", is_output_list=True),
            ],
        )

    @staticmethod
    def _resolve_pattern(glob_pattern):
        """Resolve a glob pattern to an absolute one.

        Absolute patterns matching real files are used as-is. Patterns with a
        leading slash that match nothing (e.g. "/input/3d/*.ply", meant
        relative to the ComfyUI base) retry against the base with the slash
        stripped. Relative patterns resolve against ComfyUI's input directory
        (same convention as CAD_Load_From_Glob).
        """
        if os.path.isabs(glob_pattern):
            if glob_module.glob(glob_pattern, recursive=True):
                return glob_pattern
            try:
                import folder_paths
                base = getattr(folder_paths, "base_path", None)
            except (ImportError, AttributeError):
                base = os.environ.get("COMFYUI_BASE")
            if base:
                rebased = os.path.join(base, glob_pattern.lstrip("/\\"))
                if glob_module.glob(rebased, recursive=True):
                    return rebased
            return glob_pattern
        if COMFYUI_INPUT_FOLDER is not None:
            return os.path.join(COMFYUI_INPUT_FOLDER, glob_pattern)
        return glob_pattern

    @classmethod
    def fingerprint_inputs(cls, glob_pattern, sort_by="name", start_index=0, max_meshes=-1, num_workers=1):
        """Force re-execution when any matched file changes."""
        pattern = cls._resolve_pattern(glob_pattern)
        matched_files = glob_module.glob(pattern, recursive=True)
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
    def _load_one(cls, path):
        """Load a single mesh file. Returns (mesh, texture, path), or None on failure."""
        try:
            log.info("Loading: %s", path)
            mesh, error = mesh_io.load_mesh_file(path)

            if mesh is None:
                log.warning("Failed to load %s: %s", path, error)
                return None

            # Handle both meshes and pointclouds
            if hasattr(mesh, 'faces') and mesh.faces is not None:
                log.info("Loaded: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
            else:
                log.info("Loaded pointcloud: %d points", len(mesh.vertices))

            return (mesh, cls._extract_texture_image(mesh), path)

        except Exception as e:
            log.error("Error loading %s: %s", path, e)
            return None

    @classmethod
    def execute(cls, glob_pattern, sort_by="name", start_index=0, max_meshes=-1, num_workers=1):
        """
        Load meshes matching the glob pattern.

        Args:
            glob_pattern: Glob pattern to match mesh files
            sort_by: How to sort matched files ("name" or "modified_time")
            start_index: Skip the first N matched files
            max_meshes: Load up to N files after start_index (-1 = unlimited)
            num_workers: Number of files to load concurrently (1 = sequential)

        Returns:
            tuple: (list of trimesh.Trimesh, list of IMAGE, list of file paths)
        """
        if not glob_pattern or glob_pattern.strip() == "":
            raise ValueError("Glob pattern cannot be empty")

        glob_pattern = glob_pattern.strip()
        pattern = cls._resolve_pattern(glob_pattern)

        # Find matching files
        matched_files = glob_module.glob(pattern, recursive=True)

        if not matched_files:
            log.warning("No files matched pattern: %s", pattern)
            return io.NodeOutput([], [], [])

        # Sort files
        if sort_by == "name":
            matched_files.sort()
        else:
            matched_files.sort(key=os.path.getmtime)

        if start_index > 0:
            if start_index >= len(matched_files):
                raise ValueError(
                    f"start_index ({start_index}) is >= number of matched files ({len(matched_files)})"
                )
            matched_files = matched_files[start_index:]

        if max_meshes > 0:
            matched_files = matched_files[:max_meshes]

        log.info("Found %d files matching pattern (after start_index/max_meshes)", len(matched_files))

        if num_workers > 1 and len(matched_files) > 1:
            from concurrent.futures import ThreadPoolExecutor
            log.info("Loading with %d workers", num_workers)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # executor.map preserves input order in its results, even though
                # the underlying loads complete out of order.
                results = list(executor.map(cls._load_one, matched_files))
        else:
            results = [cls._load_one(path) for path in matched_files]

        meshes = []
        textures = []
        file_paths = []
        for result in results:
            if result is None:
                continue
            mesh, texture, path = result
            meshes.append(mesh)
            textures.append(texture)
            file_paths.append(path)

        log.info("Successfully loaded %d mesh(es)", len(meshes))
        return io.NodeOutput(meshes, textures, file_paths)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshGlob": LoadMeshGlob,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshGlob": "Load Mesh Batch (Glob)",
}
