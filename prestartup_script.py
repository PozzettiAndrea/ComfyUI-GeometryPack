"""ComfyUI-GeometryPack Prestartup Script."""

import logging
from pathlib import Path

from comfy_env import setup_env, copy_files

log = logging.getLogger("geometrypack")

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy input assets into the CONFIGURED input directory, not the code-tree one:
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
