"""
GeometryPack PreStartup Script
Copies the Stanford Bunny example mesh to ComfyUI's input folder on startup.
"""
import os
import shutil

def setup_example_assets():
    """Copy Stanford Bunny STL to ComfyUI input folder if not already present."""
    try:
        import folder_paths

        # Get paths
        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(custom_node_dir, "assets", "Stanford_Bunny.stl")
        dest_file = os.path.join(input_folder, "Stanford_Bunny.stl")

        # Copy if source exists and destination doesn't
        if os.path.exists(source_file):
            if not os.path.exists(dest_file):
                shutil.copy2(source_file, dest_file)
                print(f"[GeometryPack] Copied Stanford_Bunny.stl to input folder")
            else:
                print(f"[GeometryPack] Stanford_Bunny.stl already exists in input folder")
        else:
            print(f"[GeometryPack] Warning: assets/Stanford_Bunny.stl not found")

    except Exception as e:
        print(f"[GeometryPack] Error setting up example assets: {e}")

# Run on import
setup_example_assets()
