# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack PreStartup Script
- Sets up LD_LIBRARY_PATH for pixi conda libraries
- Copies 3D viewer infrastructure from comfy-3d-viewers package
- Generates dynamic widget visibility mappings
- Copies example 3D assets to ComfyUI input folder on startup
"""
# Must run before any native library imports
from comfy_env import setup_env
setup_env()

import json
import os
import shutil


def copy_3d_viewers():
    """Copy 3D viewer files from comfy-3d-viewers package to web/ directory."""
    try:
        from comfy_3d_viewers import get_js_dir, get_html_dir, get_utils_dir, get_nodes_dir, get_assets_dir

        custom_node_dir = os.path.dirname(os.path.abspath(__file__))
        web_js_dir = os.path.join(custom_node_dir, "web", "js")
        web_dir = os.path.join(custom_node_dir, "web")

        # Ensure directories exist
        os.makedirs(web_js_dir, exist_ok=True)
        os.makedirs(os.path.join(web_js_dir, "utils"), exist_ok=True)
        os.makedirs(os.path.join(web_js_dir, "viewer"), exist_ok=True)

        copied_count = 0

        # Copy JS bundles (vtk-gltf.js, gsplat-bundle.js, viewer-bundle.js)
        src_js_dir = get_js_dir()
        for filename in os.listdir(src_js_dir):
            if filename.endswith('.js') and not os.path.isdir(os.path.join(src_js_dir, filename)):
                src = os.path.join(src_js_dir, filename)
                dst = os.path.join(web_js_dir, filename)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1

        # Copy utils directory
        src_utils_dir = get_utils_dir()
        if os.path.exists(src_utils_dir):
            dst_utils_dir = os.path.join(web_js_dir, "utils")
            for filename in os.listdir(src_utils_dir):
                if filename.endswith('.js'):
                    src = os.path.join(src_utils_dir, filename)
                    dst = os.path.join(dst_utils_dir, filename)
                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        copied_count += 1

        # Copy viewer directory (modular viewer source)
        src_viewer_dir = os.path.join(src_js_dir, "viewer")
        if os.path.exists(src_viewer_dir):
            dst_viewer_dir = os.path.join(web_js_dir, "viewer")
            for root, dirs, files in os.walk(src_viewer_dir):
                rel_path = os.path.relpath(root, src_viewer_dir)
                dst_root = os.path.join(dst_viewer_dir, rel_path) if rel_path != '.' else dst_viewer_dir
                os.makedirs(dst_root, exist_ok=True)
                for filename in files:
                    if filename.endswith('.js') or filename.endswith('.css'):
                        src = os.path.join(root, filename)
                        dst = os.path.join(dst_root, filename)
                        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                            shutil.copy2(src, dst)
                            copied_count += 1

        # Copy HTML viewer templates
        src_html_dir = get_html_dir()
        if os.path.exists(src_html_dir):
            for filename in os.listdir(src_html_dir):
                if filename.endswith('.html'):
                    src = os.path.join(src_html_dir, filename)
                    dst = os.path.join(web_dir, filename)
                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        copied_count += 1

        # Copy node widget JS files
        src_nodes_dir = get_nodes_dir()
        if os.path.exists(src_nodes_dir):
            for filename in os.listdir(src_nodes_dir):
                if filename.endswith('.js'):
                    src = os.path.join(src_nodes_dir, filename)
                    dst = os.path.join(web_js_dir, filename)
                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        copied_count += 1

        # Copy assets (HDR environments, etc.)
        src_assets_dir = get_assets_dir()
        if os.path.exists(src_assets_dir):
            dst_assets_dir = os.path.join(web_dir, "assets")
            os.makedirs(dst_assets_dir, exist_ok=True)
            for filename in os.listdir(src_assets_dir):
                src = os.path.join(src_assets_dir, filename)
                dst = os.path.join(dst_assets_dir, filename)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1

        if copied_count > 0:
            print(f"[GeometryPack] Copied {copied_count} file(s) from comfy-3d-viewers package")

    except ImportError:
        print("[GeometryPack] Warning: comfy-3d-viewers not installed, 3D viewers may not work")
    except Exception as e:
        print(f"[GeometryPack] Error copying 3D viewers: {e}")


def copy_example_assets():
    """Copy all files and folders from assets/ directory to ComfyUI input/3d directory."""
    try:
        import folder_paths

        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # Create input/3d subdirectory
        input_3d_folder = os.path.join(input_folder, "3d")
        os.makedirs(input_3d_folder, exist_ok=True)

        # Copy entire assets/ folder structure
        assets_folder = os.path.join(custom_node_dir, "assets")
        if not os.path.exists(assets_folder):
            print(f"[GeometryPack] Warning: assets folder not found at {assets_folder}")
            return

        copied_count = 0
        for root, dirs, files in os.walk(assets_folder):
            # Calculate relative path from assets folder
            rel_path = os.path.relpath(root, assets_folder)

            # Create corresponding subdirectory in destination
            if rel_path != '.':
                dest_dir = os.path.join(input_3d_folder, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = input_3d_folder

            # Copy files
            for file in files:
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    # Show relative path for clarity
                    rel_dest = os.path.join(rel_path, file) if rel_path != '.' else file
                    print(f"[GeometryPack] Copied {rel_dest} to input/3d/")

        if copied_count > 0:
            print(f"[GeometryPack] [OK] Copied {copied_count} asset(s) to {input_3d_folder}")
        else:
            print(f"[GeometryPack] All assets already exist in {input_3d_folder}")

    except Exception as e:
        print(f"[GeometryPack] Error copying assets: {e}")


def copy_dynamic_widgets_js():
    """Copy dynamic_widgets.js from pip package to web/js/."""
    try:
        from comfy_dynamic_widgets import get_js_path

        source = get_js_path()
        if not os.path.exists(source):
            print(f"[GeometryPack] Warning: dynamic_widgets.js not found at {source}")
            return

        custom_node_dir = os.path.dirname(os.path.abspath(__file__))
        dest = os.path.join(custom_node_dir, "web", "js", "dynamic_widgets.js")

        # Ensure web/js directory exists
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Copy if source is newer or dest doesn't exist
        if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
            shutil.copy2(source, dest)
            print("[GeometryPack] Copied dynamic_widgets.js from comfy-dynamic-widgets package")

    except ImportError:
        print("[GeometryPack] Warning: comfy-dynamic-widgets not installed")
    except Exception as e:
        print(f"[GeometryPack] Error copying JS: {e}")


def generate_widget_mappings():
    """Generate dynamic widget visibility mappings using comfy-dynamic-widgets."""
    try:
        from comfy_dynamic_widgets import scan_all_nodes, generate_mappings

        # Scan all nodes for visible_when metadata
        configs = scan_all_nodes()

        if not configs:
            print("[GeometryPack] No visible_when metadata found in any nodes")
            return

        # Generate JS-friendly mappings
        mappings = generate_mappings(configs)

        # Write to web/js/mappings.json
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(custom_node_dir, "web", "js", "mappings.json")

        with open(output_path, "w") as f:
            json.dump(mappings, f, indent=2)

        node_count = len(configs)
        selector_count = sum(len(c.get("selectors", {})) for c in configs.values())
        print(f"[GeometryPack] Generated widget mappings for {node_count} nodes ({selector_count} selectors)")

    except ImportError:
        print("[GeometryPack] Warning: comfy-dynamic-widgets not installed, skipping widget mappings")
    except Exception as e:
        print(f"[GeometryPack] Error generating widget mappings: {e}")


# Run on import
copy_3d_viewers()
copy_dynamic_widgets_js()
# Note: generate_widget_mappings() moved to __init__.py where NODE_CLASS_MAPPINGS is available
copy_example_assets()
