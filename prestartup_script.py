"""ComfyUI-GeometryPack Prestartup Script."""

import logging
import os
import sys

log = logging.getLogger("geometrypack")

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy viewers (GeometryPack uses many viewer types)
viewers = [
    "viewer", "vtk", "vtk_batch", "vtk_textured", "pointcloud_vtk",
    "multi", "multi_slider", "dual", "dual_slider", "dual_textured",
    "uv", "pbr", "gaussian",
    "fbx", "fbx_debug", "fbx_compare",
    "bvh", "fbx_animation", "compare_smpl_bvh",
    "text_report",
    "ultimate_inspection",
    "slicer",
    "warp_mesh",
]
for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        log.warning("Failed to copy viewer %s: %s", viewer, e)

# Copy GeometryPack-local frontend JS (tracked source -> served web/js, since web/ is generated)
copy_files(SCRIPT_DIR / "web_src", SCRIPT_DIR / "web" / "js", "*.js")

# Copy assets into the CONFIGURED input directory, not the code-tree one:
# ComfyUI Desktop separates user data from the code tree (--base-directory),
# so COMFYUI_DIR/input is never scanned by the load nodes there. main.py
# calls apply_custom_paths() before prestartup scripts, so folder_paths is
# already configured here.
try:
    import folder_paths
    INPUT_DIR = Path(folder_paths.get_input_directory())
except Exception:
    INPUT_DIR = COMFYUI_DIR / "input"
copy_files(SCRIPT_DIR / "assets", INPUT_DIR / "3d", "**/*")

copy_files(SCRIPT_DIR / "assets", INPUT_DIR, "*.exr")