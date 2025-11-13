"""
GeometryPack PreStartup Script
Copies example 3D assets and workflows to ComfyUI folders on startup.
"""
import os
import shutil

def setup_example_assets():
    """Copy example 3D assets to ComfyUI input/3d folder and workflows to user folder."""
    try:
        import folder_paths

        # Get paths
        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # Create input/3d subdirectory if it doesn't exist
        input_3d_folder = os.path.join(input_folder, "3d")
        os.makedirs(input_3d_folder, exist_ok=True)

        # Define all example assets to copy
        example_assets = [
            "Stanford_Bunny.stl",
            "spot_the_cow.obj",
            "utah_teapot.stl"
        ]

        # Copy each asset to input/3d/
        assets_folder = os.path.join(custom_node_dir, "assets")
        for asset_file in example_assets:
            source_file = os.path.join(assets_folder, asset_file)
            dest_file = os.path.join(input_3d_folder, asset_file)

            if os.path.exists(source_file):
                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    print(f"[GeometryPack] Copied {asset_file} to input/3d/")
            else:
                print(f"[GeometryPack] Warning: assets/{asset_file} not found")

        # Copy workflows with GeometryPack- prefix
        workflows_src = os.path.join(custom_node_dir, "workflows")
        if os.path.exists(workflows_src):
            # Get ComfyUI root directory (parent of input folder)
            comfyui_root = os.path.dirname(input_folder)
            # Check for user/default/workflows or fallback to root
            user_workflows_dir = os.path.join(comfyui_root, "user", "default", "workflows")
            if not os.path.exists(user_workflows_dir):
                user_workflows_dir = comfyui_root

            # Copy each workflow file with prefix
            for workflow_file in os.listdir(workflows_src):
                if workflow_file.endswith('.json'):
                    source_workflow = os.path.join(workflows_src, workflow_file)
                    dest_workflow = os.path.join(user_workflows_dir, f"GeometryPack-{workflow_file}")

                    if not os.path.exists(dest_workflow):
                        shutil.copy2(source_workflow, dest_workflow)
                        print(f"[GeometryPack] Copied workflow GeometryPack-{workflow_file}")

    except Exception as e:
        print(f"[GeometryPack] Error setting up example assets: {e}")

# Run on import
setup_example_assets()
