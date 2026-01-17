# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack PreStartup Script
- Generates dynamic widget visibility mappings
- Copies example 3D assets to ComfyUI input folder on startup
"""
import json
import os
import shutil


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
generate_widget_mappings()
copy_example_assets()
