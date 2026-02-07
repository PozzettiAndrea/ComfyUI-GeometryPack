"""ComfyUI-GeometryPack Prestartup Script."""

import os
import sys

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

# --- Diagnostic hooks (active when COMFY_ENV_DEBUG=1) ---
if os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes"):
    import atexit

    def _exit_hook():
        print("[geompack] atexit handler called — clean Python shutdown",
              file=sys.stderr, flush=True)
    atexit.register(_exit_hook)

    _orig_excepthook = sys.excepthook
    def _crash_excepthook(exc_type, exc_value, exc_tb):
        print(f"[geompack] UNHANDLED: {exc_type.__name__}: {exc_value}",
              file=sys.stderr, flush=True)
        _orig_excepthook(exc_type, exc_value, exc_tb)
    sys.excepthook = _crash_excepthook

    class _ImportTracer:
        """Log first-time imports of key modules to identify crash location."""
        _PREFIXES = ('comfy', 'execution', 'server', 'protocol', 'nodes',
                     'torch', 'comfyui_version', 'hook_breaker', 'comfy_aimdo',
                     'app', 'cuda_malloc')

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in sys.modules:
                parts = fullname.split('.')
                if parts[0] in self._PREFIXES and len(parts) <= 2:
                    print(f"[import-trace] {fullname}",
                          file=sys.stderr, flush=True)
            return None

    sys.meta_path.insert(0, _ImportTracer())

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy viewers (GeometryPack uses many viewer types)
viewers = [
    "viewer", "vtk", "vtk_textured", "pointcloud_vtk",
    "multi", "dual", "dual_slider", "dual_textured",
    "uv", "pbr", "gaussian",
    "fbx", "fbx_debug", "fbx_compare",
    "bvh", "fbx_animation", "compare_smpl_bvh"
]
for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        print(e)

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

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*")
