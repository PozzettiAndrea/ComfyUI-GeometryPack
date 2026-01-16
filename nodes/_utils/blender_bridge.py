# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
DEPRECATED: Shared Blender utilities for ComfyUI-GeometryPack nodes.

This module is DEPRECATED. Blender operations now use bpy_bridge.py with
comfy-env isolation, which provides direct bpy API calls without subprocess
or temp file I/O.

Use bpy_bridge.py instead:
    from .._utils.bpy_bridge import bpy_smart_uv_project, bpy_voxel_remesh, etc.

This module is kept for backwards compatibility but will be removed in a
future version.
"""

import warnings
warnings.warn(
    "blender_bridge.py is DEPRECATED. "
    "Blender operations now use bpy_bridge.py with comfy-env isolation. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import trimesh as trimesh_module


def find_blender():
    """
    Find Blender executable on the system.

    Checks in order:
    1. Local installation in _blender/ (downloaded by install.py)
    2. System installation (PATH or common locations)

    Returns:
        str: Path to Blender executable

    Raises:
        RuntimeError: If Blender not found
    """
    # Get the directory containing this file
    current_dir = Path(__file__).parent.parent.parent  # Go up from nodes/_utils/ to package root
    local_blender_dir = current_dir / "_blender"

    # First, check for local Blender installation
    if local_blender_dir.exists():
        # Search for blender executable in _blender/
        blender_executables = []

        # Windows
        blender_executables.extend(list(local_blender_dir.rglob("blender.exe")))

        # Linux/macOS
        blender_executables.extend([
            p for p in local_blender_dir.rglob("blender")
            if p.is_file() and os.access(p, os.X_OK)
        ])

        if blender_executables:
            blender_path = str(blender_executables[0])
            print(f"[Blender] Using local Blender: {blender_path}")
            return blender_path

    # Fall back to system installation
    common_paths = [
        'blender',  # In PATH
        '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
        'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',  # Windows
        '/usr/bin/blender',  # Linux
        '/usr/local/bin/blender',  # Linux
    ]

    for path in common_paths:
        if shutil.which(path) or os.path.exists(path):
            print(f"[Blender] Found system Blender: {path}")
            return path

    raise RuntimeError(
        "Blender not found. Please run 'python blender_install.py' to download Blender,\n"
        "or install it manually from: https://www.blender.org/download/"
    )


def run_blender_script(script, timeout=300, capture_output=True):
    """
    Run a Python script in Blender's background mode.

    Args:
        script: Python script string to execute in Blender
        timeout: Maximum execution time in seconds (default: 300)
        capture_output: Whether to capture stdout/stderr (default: True)

    Returns:
        subprocess.CompletedProcess: Result of the subprocess call

    Raises:
        RuntimeError: If Blender execution fails
    """
    blender_path = find_blender()

    result = subprocess.run(
        [blender_path, '--background', '--python-expr', script],
        capture_output=capture_output,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Blender execution failed: {result.stderr}")

    return result


def run_blender_mesh_operation(input_mesh, blender_script_template,
                                output_format='obj', timeout=300,
                                preserve_metadata=True, metadata_key='blender_operation',
                                metadata_values=None):
    """
    Execute a Blender mesh operation with automatic temp file management.

    Args:
        input_mesh: Input trimesh.Trimesh object
        blender_script_template: Script template with {input_path} and {output_path} placeholders
        output_format: Output file format ('obj', 'ply', etc.)
        timeout: Maximum execution time in seconds
        preserve_metadata: Whether to copy metadata from input to output
        metadata_key: Key to store operation metadata under
        metadata_values: Dictionary of metadata to store

    Returns:
        trimesh.Trimesh: Resulting mesh after Blender operation

    Raises:
        RuntimeError: If operation fails
    """
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
        input_path = f_in.name
        input_mesh.export(input_path)

    with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as f_out:
        output_path = f_out.name

    try:
        # Format the script with file paths
        script = blender_script_template.format(
            input_path=input_path,
            output_path=output_path
        )

        # Run Blender
        run_blender_script(script, timeout=timeout)

        # Load the result
        result_mesh = trimesh_module.load(output_path, process=False)

        # Handle Scene objects
        if isinstance(result_mesh, trimesh_module.Scene):
            result_mesh = result_mesh.dump(concatenate=True)

        # Preserve metadata
        if preserve_metadata:
            result_mesh.metadata = input_mesh.metadata.copy()

        # Add operation metadata
        if metadata_values:
            result_mesh.metadata[metadata_key] = metadata_values

        return result_mesh

    finally:
        # Cleanup temp files
        cleanup_temp_files([input_path, output_path])


def cleanup_temp_files(file_paths):
    """
    Clean up temporary files.

    Args:
        file_paths: List of file paths to remove
    """
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                print(f"[Blender] Warning: Could not remove temp file {path}: {e}")


# Common Blender script templates
BLENDER_IMPORT_OBJ = """
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
"""

BLENDER_EXPORT_OBJ = """
# Export result
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

BLENDER_EXPORT_OBJ_WITH_UV = """
# Export result with UVs
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=True,
    export_materials=False
)
"""
