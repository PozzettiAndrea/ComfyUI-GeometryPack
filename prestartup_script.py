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
]
for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        log.warning("Failed to copy viewer %s: %s", viewer, e)

# Copy dynamic widgets
try:
    from comfy_dynamic_widgets import get_js_path
    import shutil
    src = Path(get_js_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "dynamic_widgets.js"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy GeometryPack-local frontend JS (tracked source -> served web/js, since web/ is generated)
copy_files(SCRIPT_DIR / "web_src", SCRIPT_DIR / "web" / "js", "*.js")

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*")
