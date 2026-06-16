# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh Batch Node - Load multiple meshes from a folder (batch loading)
"""

import logging
import os

log = logging.getLogger("geometrypack")

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
    # Get ComfyUI root (parent of input/output folders)
    COMFYUI_ROOT = os.path.dirname(COMFYUI_INPUT_FOLDER)
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None
    COMFYUI_ROOT = None

from . import mesh_io
from comfy_api.latest import io


def _n_faces(m):
    f = getattr(m, "faces", None)
    return len(f) if f is not None else 0


def _load_mesh_worker(file_path):
    """Top-level worker (picklable) for the process pool: load one mesh file.
    Returns the trimesh (or None on failure). True CPU parallelism for parsing,
    unlike threads (trimesh's OBJ parser is largely GIL-bound Python)."""
    try:
        m, _err = mesh_io.load_mesh_file(file_path)
        return m
    except Exception:
        return None


class LoadMeshBatch(io.ComfyNode):
    """
    Load multiple meshes from a folder (batch loading).
    Similar to ComfyUI's image batch loading, with start_index and max_meshes controls.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackLoadMeshBatch",
            display_name="Load Mesh Batch",
            category="geompack/io",
            inputs=[
                io.String.Input("folder_path", default="3d", multiline=False),
                io.Int.Input("start_index", default=0, min=0, max=100000),
                io.Int.Input("max_meshes", default=-1, min=-1, max=100000),
                io.Boolean.Input("use_multithreading", default=True,
                    tooltip="Load files in parallel across CPU cores using a process pool "
                            "(real parallelism for the CPU-bound OBJ parsing; falls back to "
                            "serial on error)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    # Supported mesh file extensions
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']


    @classmethod
    def execute(cls, folder_path, start_index, max_meshes, use_multithreading=True):
        """
        Load multiple meshes from a folder.

        Args:
            folder_path: Path to folder containing mesh files (relative to input folder or absolute)
            start_index: Skip first N meshes (0 = start from beginning)
            max_meshes: Load up to N meshes (-1 = unlimited)

        Returns:
            tuple: (list of trimesh.Trimesh objects,)
        """
        if not folder_path or folder_path.strip() == "":
            raise ValueError("Folder path cannot be empty")

        # Resolve folder path - check multiple locations
        # Order: ComfyUI root (for paths like "output/folder"), input folder, output folder, absolute
        full_folder_path = None
        searched_paths = []

        # 1. Try relative to ComfyUI root (handles "output/mesh_output", "input/3d", etc.)
        if COMFYUI_ROOT is not None:
            root_path = os.path.join(COMFYUI_ROOT, folder_path)
            searched_paths.append(f"{root_path} (ComfyUI root)")
            if os.path.exists(root_path) and os.path.isdir(root_path):
                full_folder_path = root_path
                log.info("Found folder relative to ComfyUI root: %s", folder_path)

        # 2. Try in ComfyUI input folder
        if full_folder_path is None and COMFYUI_INPUT_FOLDER is not None:
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, folder_path)
            searched_paths.append(f"{input_path} (input folder)")
            if os.path.exists(input_path) and os.path.isdir(input_path):
                full_folder_path = input_path
                log.info("Found folder in input: %s", folder_path)

        # 3. Try in ComfyUI output folder
        if full_folder_path is None and COMFYUI_OUTPUT_FOLDER is not None:
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, folder_path)
            searched_paths.append(f"{output_path} (output folder)")
            if os.path.exists(output_path) and os.path.isdir(output_path):
                full_folder_path = output_path
                log.info("Found folder in output: %s", folder_path)

        # 4. Try as absolute path
        if full_folder_path is None:
            searched_paths.append(f"{folder_path} (absolute)")
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                full_folder_path = folder_path
                log.info("Using absolute path: %s", folder_path)
            else:
                error_msg = f"Folder not found: '{folder_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Scan folder for mesh files
        mesh_files = []
        for filename in os.listdir(full_folder_path):
            file_lower = filename.lower()
            if any(file_lower.endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                mesh_files.append(filename)

        # Sort files alphabetically for consistent ordering
        mesh_files.sort()

        if len(mesh_files) == 0:
            raise ValueError(f"No mesh files found in folder: {full_folder_path}\n"
                           f"Supported extensions: {', '.join(cls.SUPPORTED_EXTENSIONS)}")

        log.info("Found %d mesh files", len(mesh_files))

        # Apply start_index and max_meshes
        if start_index > 0:
            if start_index >= len(mesh_files):
                raise ValueError(f"start_index ({start_index}) is >= number of mesh files ({len(mesh_files)})")
            mesh_files = mesh_files[start_index:]
            log.info("Skipping first %d files", start_index)

        if max_meshes > 0:
            mesh_files = mesh_files[:max_meshes]
            log.info("Loading up to %d meshes", max_meshes)

        # Load all meshes, optionally in parallel. trimesh's OBJ parser is mostly
        # GIL-bound Python, so threads barely help -- use a PROCESS pool for real
        # CPU parallelism (the parse cost dwarfs the pickle-back of the result).
        total = len(mesh_files)
        paths = [os.path.join(full_folder_path, f) for f in mesh_files]

        def _record(i, filename, m, loaded):
            if m is not None:
                loaded.append(m)
                log.info("[%d/%d] Loaded %s: %d vertices, %d faces",
                         i, total, filename, len(m.vertices), _n_faces(m))
            else:
                log.warning("[%d/%d] Failed to load %s", i, total, filename)

        # ComfyUI UI progress bar (lazy import so it never breaks the metadata scan)
        def _mk_pbar():
            try:
                from comfy.utils import ProgressBar
                return ProgressBar(total)
            except Exception:
                return None

        loaded_meshes = []
        used_parallel = False
        if use_multithreading and total > 1:
            from concurrent.futures import ProcessPoolExecutor
            workers = min(total, (os.cpu_count() or 4))
            log.info("Loading %d meshes across %d processes", total, workers)
            pbar = _mk_pbar()
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for i, m in enumerate(ex.map(_load_mesh_worker, paths), 1):
                        _record(i, mesh_files[i - 1], m, loaded_meshes)
                        if pbar is not None:
                            pbar.update(1)
                used_parallel = True
            except Exception as e:
                log.warning("Parallel load failed (%s); falling back to serial", e)
                loaded_meshes = []

        if not used_parallel:
            pbar = _mk_pbar()
            for i, (filename, path) in enumerate(zip(mesh_files, paths), 1):
                _record(i, filename, _load_mesh_worker(path), loaded_meshes)
                if pbar is not None:
                    pbar.update(1)

        if len(loaded_meshes) == 0:
            raise ValueError(f"Failed to load any meshes from folder: {full_folder_path}")

        log.info("Successfully loaded %d meshes", len(loaded_meshes))

        return io.NodeOutput(loaded_meshes)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshBatch": LoadMeshBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshBatch": "Load Mesh Batch",
}
