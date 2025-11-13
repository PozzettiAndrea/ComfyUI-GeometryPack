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

    # Supported mesh extensions for file browser
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available mesh files (like LoadImage does)
        mesh_files = cls.get_mesh_files()

        # If no files found, provide a default empty list
        if not mesh_files:
            mesh_files = ["No mesh files found in input/3d or input folders"]

        return {
            "required": {
                "file_path": (mesh_files, ),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "geompack/io"

    @classmethod
    def get_mesh_files(cls):
        """Get list of available mesh files in input/3d and input folders."""
        mesh_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        return sorted(mesh_files)

    @classmethod
    def IS_CHANGED(cls, file_path):
        """Force re-execution when file changes."""
        if COMFYUI_INPUT_FOLDER is not None:
            # Check file modification time
            full_path = None
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)

            if os.path.exists(input_3d_path):
                full_path = input_3d_path
            elif os.path.exists(input_path):
                full_path = input_path

            if full_path and os.path.exists(full_path):
                return os.path.getmtime(full_path)

        return file_path

    def load_mesh(self, file_path):
        """
        Load mesh from file.

        Looks for files in ComfyUI's input/3d folder first, then input folder, then tries absolute path.

        Args:
            file_path: Path to mesh file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[LoadMesh] Found mesh in input/3d folder: {file_path}")

            # Second, try in ComfyUI input folder (for backward compatibility)
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[LoadMesh] Found mesh in input folder: {file_path}")

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[LoadMesh] Loading from absolute path: {file_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load the mesh
        loaded_mesh, error = mesh_utils.load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        print(f"[LoadMesh] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh,)


class LoadMeshBatch:
    """
    Load multiple meshes from a folder (batch loading).
    Similar to ComfyUI's image batch loading, with start_index and max_meshes controls.
    """

    # Supported mesh file extensions
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "3d",
                    "multiline": False
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000
                }),
                "max_meshes": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("meshes",)
    FUNCTION = "load_mesh_batch"
    CATEGORY = "geompack/io"
    OUTPUT_IS_LIST = (True,)

    def load_mesh_batch(self, folder_path, start_index, max_meshes):
        """
        Load multiple meshes from a folder.

        Args:
            folder_path: Path to folder containing mesh files (relative to input folder or absolute)
            start_index: Skip first N meshes (0 = start from beginning)
            max_meshes: Load up to N meshes (-1 = unlimited)

        Returns:
            tuple: (list of trimesh.Trimesh objects,)
        """
        if not folder_path or folder_path.strip() == "":
            raise ValueError("Folder path cannot be empty")

        # Resolve folder path
        full_folder_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Try in ComfyUI input folder
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, folder_path)
            searched_paths.append(input_path)
            if os.path.exists(input_path) and os.path.isdir(input_path):
                full_folder_path = input_path
                print(f"[LoadMeshBatch] Found folder in input: {folder_path}")

        # If not found, try as absolute path
        if full_folder_path is None:
            searched_paths.append(folder_path)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                full_folder_path = folder_path
                print(f"[LoadMeshBatch] Using absolute path: {folder_path}")
            else:
                error_msg = f"Folder not found: '{folder_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Scan folder for mesh files
        mesh_files = []
        for filename in os.listdir(full_folder_path):
            file_lower = filename.lower()
            if any(file_lower.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                mesh_files.append(filename)

        # Sort files alphabetically for consistent ordering
        mesh_files.sort()

        if len(mesh_files) == 0:
            raise ValueError(f"No mesh files found in folder: {full_folder_path}\n"
                           f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        print(f"[LoadMeshBatch] Found {len(mesh_files)} mesh files")

        # Apply start_index and max_meshes
        if start_index > 0:
            if start_index >= len(mesh_files):
                raise ValueError(f"start_index ({start_index}) is >= number of mesh files ({len(mesh_files)})")
            mesh_files = mesh_files[start_index:]
            print(f"[LoadMeshBatch] Skipping first {start_index} files")

        if max_meshes > 0:
            mesh_files = mesh_files[:max_meshes]
            print(f"[LoadMeshBatch] Loading up to {max_meshes} meshes")

        # Load all meshes
        loaded_meshes = []
        for i, filename in enumerate(mesh_files):
            file_path = os.path.join(full_folder_path, filename)
            try:
                loaded_mesh, error = mesh_utils.load_mesh_file(file_path)
                if loaded_mesh is None:
                    print(f"[LoadMeshBatch] Warning: Failed to load {filename}: {error}")
                    continue

                loaded_meshes.append(loaded_mesh)
                print(f"[LoadMeshBatch] [{i+1}/{len(mesh_files)}] Loaded {filename}: "
                      f"{len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")
            except Exception as e:
                print(f"[LoadMeshBatch] Warning: Error loading {filename}: {e}")
                continue

        if len(loaded_meshes) == 0:
            raise ValueError(f"Failed to load any meshes from folder: {full_folder_path}")

        print(f"[LoadMeshBatch] Successfully loaded {len(loaded_meshes)} meshes")

        return (loaded_meshes,)


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

        # Debug: Check what we received
        print(f"[SaveMesh] Received mesh type: {type(trimesh)}")
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None instead of a mesh object. Check that the previous node is outputting a mesh.")

        # Check if mesh has data
        try:
            vertex_count = len(trimesh.vertices) if hasattr(trimesh, 'vertices') else 0
            face_count = len(trimesh.faces) if hasattr(trimesh, 'faces') else 0
            print(f"[SaveMesh] Mesh has {vertex_count} vertices, {face_count} faces")

            if vertex_count == 0 or face_count == 0:
                raise ValueError(
                    f"Cannot save empty mesh (vertices: {vertex_count}, faces: {face_count}). "
                    "Check that the previous node is producing valid geometry."
                )
        except Exception as e:
            raise ValueError(f"Error checking mesh properties: {e}. Received object may not be a valid mesh.")

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
    "GeomPackLoadMeshBatch": LoadMeshBatch,
    "GeomPackSaveMesh": SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMesh": "Load Mesh",
    "GeomPackLoadMeshBatch": "Load Mesh Batch",
    "GeomPackSaveMesh": "Save Mesh",
}
