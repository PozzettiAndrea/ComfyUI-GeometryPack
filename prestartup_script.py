"""ComfyUI-GeometryPack Prestartup Script."""

from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy 3D viewer files from comfy-3d-viewers package
try:
    from comfy_3d_viewers import (
        get_js_dir, get_html_dir, get_utils_dir, get_nodes_dir, get_assets_dir
    )
    import shutil

    web_dir = SCRIPT_DIR / "web"
    web_js_dir = web_dir / "js"
    web_js_dir.mkdir(parents=True, exist_ok=True)

    # HTML files GeometryPack actually uses (NOT fbx viewer variants)
    html_files = [
        "viewer.html", "viewer_vtk.html", "viewer_vtk_textured.html",
        "viewer_multi.html", "viewer_dual.html", "viewer_dual_slider.html",
        "viewer_dual_textured.html", "viewer_uv.html", "viewer_pbr.html",
        "viewer_gaussian.html", "viewer_bvh.html", "viewer_fbx_animation.html",
        "viewer_compare_smpl_bvh.html",
        "viewer_cad_analysis.html", "viewer_cad_curve.html", "viewer_cad_edge.html",
        "viewer_cad_edge_detail.html", "viewer_cad_edge_vtk.html",
        "viewer_cad_hierarchy.html", "viewer_cad_occ.html", "viewer_cad_roi.html",
        "viewer_cad_spline.html",
    ]
    html_dir = Path(get_html_dir())
    for name in html_files:
        src = html_dir / name
        if src.exists():
            dst = web_dir / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)

    # JS bundles (vtk-gltf.js, gsplat-bundle.js, viewer-bundle.js)
    js_dir = Path(get_js_dir())
    for f in js_dir.glob("*.js"):
        if f.is_file():
            dst = web_js_dir / f.name
            if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(f, dst)

    # Utils and viewer subdirectories
    for subdir in ["utils", "viewer"]:
        src_dir = js_dir / subdir
        if src_dir.exists():
            dst_dir = web_js_dir / subdir
            dst_dir.mkdir(exist_ok=True)
            for f in src_dir.rglob("*"):
                if f.is_file() and f.suffix in (".js", ".css"):
                    rel = f.relative_to(src_dir)
                    dst = dst_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(f, dst)

    # Node widgets (skip FBX-specific ones GeometryPack doesn't use)
    skip = {"mesh_preview_fbx.js", "debug_skeleton_widget.js", "compare_skeleton_widget.js"}
    nodes_dir = Path(get_nodes_dir())
    for f in nodes_dir.glob("*.js"):
        if f.name not in skip:
            dst = web_js_dir / f.name
            if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(f, dst)

    # Assets (HDR environments)
    assets_dir = Path(get_assets_dir())
    if assets_dir.exists():
        dst_assets = web_dir / "assets"
        dst_assets.mkdir(exist_ok=True)
        for f in assets_dir.iterdir():
            if f.is_file():
                dst = dst_assets / f.name
                if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(f, dst)

except ImportError:
    print("[GeometryPack] comfy-3d-viewers not installed")

# Copy dynamic widgets JS
try:
    from comfy_dynamic_widgets import get_js_path
    import shutil
    src = Path(get_js_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "dynamic_widgets.js"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy example assets to input/3d
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*")
