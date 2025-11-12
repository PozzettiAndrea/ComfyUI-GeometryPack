"""
I/O Nodes - Load and save mesh files
"""

import os

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

from . import mesh_utils


class LoadMesh:
    """
    Load a mesh from file (OBJ, PLY, STL, OFF, etc.)
    Now returns trimesh.Trimesh objects for better mesh handling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "Stanford_Bunny.stl",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "geompack/io"

    def load_mesh(self, file_path):
        """
        Load mesh from file.

        Looks for files in ComfyUI's input folder first, then tries absolute path.

        Args:
            file_path: Path to mesh file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None

        # First, try in ComfyUI input folder
        if COMFYUI_INPUT_FOLDER is not None:
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
            if os.path.exists(input_path):
                full_path = input_path
                print(f"[LoadMesh] Found mesh in input folder: {file_path}")

        # If not found in input folder, try as absolute path
        if full_path is None:
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[LoadMesh] Loading from absolute path: {file_path}")
            else:
                # Try one more time with input folder prefix for better error message
                if COMFYUI_INPUT_FOLDER is not None:
                    raise ValueError(
                        f"File not found: '{file_path}'\n"
                        f"Searched in:\n"
                        f"  - ComfyUI input folder: {os.path.join(COMFYUI_INPUT_FOLDER, file_path)}\n"
                        f"  - Absolute path: {file_path}"
                    )
                else:
                    raise ValueError(f"File not found: {file_path}")

        # Load the mesh
        loaded_mesh, error = mesh_utils.load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        print(f"[LoadMesh] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh,)


class SaveMesh:
    """
    Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_path": ("STRING", {
                    "default": "output.obj",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_mesh"
    CATEGORY = "geompack/io"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh, file_path):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            trimesh: trimesh.Trimesh object
            file_path: Output file path (relative to output folder or absolute)

        Returns:
            tuple: (status_message,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Determine full output path
        full_path = file_path

        # If path is relative and we have output folder, use it
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            print(f"[SaveMesh] Saving to output folder: {file_path}")
        else:
            print(f"[SaveMesh] Saving to: {file_path}")

        # Save the mesh
        success, error = mesh_utils.save_mesh_file(trimesh, full_path)

        if not success:
            raise ValueError(f"Failed to save trimesh: {error}")

        status = f"Successfully saved mesh to: {full_path}\n"
        status += f"  Vertices: {len(trimesh.vertices)}\n"
        status += f"  Faces: {len(trimesh.faces)}"

        print(f"[SaveMesh] {status}")

        return (status,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMesh": LoadMesh,
    "GeomPackSaveMesh": SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMesh": "Load Mesh",
    "GeomPackSaveMesh": "Save Mesh",
}
