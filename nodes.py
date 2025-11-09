"""
ComfyUI GeomPack - Custom Nodes for Geometry Processing
Now using trimesh and CGAL for enhanced mesh processing
"""

import numpy as np
import trimesh
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

# Handle both relative and absolute imports
try:
    from . import mesh_utils
except ImportError:
    import mesh_utils


def _find_blender():
    """
    Find Blender executable on the system.

    Returns:
        str: Path to Blender executable

    Raises:
        RuntimeError: If Blender not found
    """
    # Try common locations
    common_paths = [
        'blender',  # In PATH
        '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
        'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',  # Windows
        '/usr/bin/blender',  # Linux
        '/usr/local/bin/blender',  # Linux
    ]

    for path in common_paths:
        if shutil.which(path) or os.path.exists(path):
            print(f"[Blender] Found Blender at: {path}")
            return path

    raise RuntimeError(
        "Blender not found. Please install Blender and ensure it's in your PATH.\n"
        "Download from: https://www.blender.org/download/"
    )


class ExampleLibiglNode:
    """
    Example node demonstrating the ComfyUI node structure.
    This is a cookie-cutter template - replace with actual functionality.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for this node.

        Returns:
            dict: Input type definitions with 'required' and 'optional' keys
        """
        return {
            "required": {
                "text_input": ("STRING", {
                    "default": "Hello from GeomPack!",
                    "multiline": False
                }),
                "number_input": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "float_input": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "mode": (["option1", "option2", "option3"], {
                    "default": "option1"
                }),
            },
            "optional": {
                "optional_input": ("STRING", {
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_text", "output_number", "output_float")
    FUNCTION = "execute"
    CATEGORY = "geompack/examples"
    OUTPUT_NODE = False

    def execute(self, text_input, number_input, float_input, mode, optional_input=""):
        """
        Execute the node logic.

        Args:
            text_input: String input
            number_input: Integer input
            float_input: Float input
            mode: Selected mode from combo box
            optional_input: Optional string input

        Returns:
            tuple: (output_text, output_number, output_float)
        """
        # Example processing
        result_text = f"{text_input} | Mode: {mode}"
        if optional_input:
            result_text += f" | Optional: {optional_input}"

        result_number = number_input * 2
        result_float = float_input * 1.5

        print(f"[ExampleNode] Processing: {result_text}")

        return (result_text, result_number, result_float)


class MeshInfoNode:
    """
    Display detailed mesh information and statistics.
    Now using trimesh for enhanced mesh analysis.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_mesh_info"
    CATEGORY = "geompack/analysis"

    def get_mesh_info(self, mesh):
        """
        Get information about the mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            tuple: (info_string,)
        """
        info = mesh_utils.compute_mesh_info(mesh)
        return (info,)


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
                    "default": "",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
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
        mesh, error = mesh_utils.load_mesh_file(full_path)

        if mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        print(f"[LoadMesh] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


class SaveMesh:
    """
    Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
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

    def save_mesh(self, mesh, file_path):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            mesh: trimesh.Trimesh object
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
        success, error = mesh_utils.save_mesh_file(mesh, full_path)

        if not success:
            raise ValueError(f"Failed to save mesh: {error}")

        status = f"Successfully saved mesh to: {full_path}\n"
        status += f"  Vertices: {len(mesh.vertices)}\n"
        status += f"  Faces: {len(mesh.faces)}"

        print(f"[SaveMesh] {status}")

        return (status,)


class CreatePrimitive:
    """
    Create primitive geometry (cube, sphere, plane)
    Uses trimesh creation functions for high-quality primitives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (["cube", "sphere", "plane"], {
                    "default": "cube"
                }),
                "size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "subdivisions": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "create_primitive"
    CATEGORY = "geompack/primitives"

    def create_primitive(self, shape, size, subdivisions=2):
        """
        Create a primitive mesh.

        Args:
            shape: Type of primitive (cube, sphere, plane)
            size: Size of the primitive
            subdivisions: Number of subdivisions (for sphere and plane)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if shape == "cube":
            mesh = mesh_utils.create_cube(size)
        elif shape == "sphere":
            mesh = mesh_utils.create_sphere(radius=size/2.0, subdivisions=subdivisions)
        elif shape == "plane":
            mesh = mesh_utils.create_plane(size=size, subdivisions=subdivisions)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        print(f"[CreatePrimitive] Created {shape}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


class CGALRemeshNode:
    """
    CGAL Isotropic Remeshing - Create uniform triangle meshes.

    Implements the algorithm from Botsch & Kobbelt (2004).
    This remeshing technique creates triangles with target edge length,
    resulting in more uniform mesh quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_edge_length": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "remesh"
    CATEGORY = "geompack/cgal"

    def remesh(self, mesh, target_edge_length, iterations):
        """
        Apply CGAL isotropic remeshing.

        Args:
            mesh: Input trimesh.Trimesh object
            target_edge_length: Target edge length for remeshed triangles
            iterations: Number of remeshing iterations

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[CGALRemesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[CGALRemesh] Target edge length: {target_edge_length}, Iterations: {iterations}")

        remeshed_mesh, error = mesh_utils.cgal_isotropic_remesh(
            mesh,
            target_edge_length,
            iterations
        )

        if remeshed_mesh is None:
            raise ValueError(f"Remeshing failed: {error}")

        print(f"[CGALRemesh] Output: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

        return (remeshed_mesh,)


class BlenderUVUnwrapNode:
    """
    UV Unwrap mesh using Blender's Smart UV Project.

    This node uses Blender's advanced UV unwrapping algorithm to generate
    UV coordinates for texturing. The mesh is exported with UV coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "angle_limit": ("FLOAT", {
                    "default": 66.0,
                    "min": 1.0,
                    "max": 89.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "island_margin": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/blender"

    def uv_unwrap(self, mesh, angle_limit, island_margin):
        """
        UV unwrap mesh using Blender's Smart UV Project.

        Args:
            mesh: Input trimesh.Trimesh object
            angle_limit: Angle threshold for creating seams (degrees)
            island_margin: Spacing between UV islands (0-1)

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        print(f"[BlenderUVUnwrap] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[BlenderUVUnwrap] Parameters: angle_limit={angle_limit}°, island_margin={island_margin}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for UV unwrapping
            script = f"""
import bpy
import math

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh (OBJ preserves geometry)
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Switch to edit mode and unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit={np.radians(angle_limit)},
    island_margin={island_margin},
    area_weight=0.0,
    correct_aspect=True,
    scale_to_bounds=False
)
bpy.ops.object.mode_set(mode='OBJECT')

# Export with UVs
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=True,
    export_materials=False
)
"""

            print(f"[BlenderUVUnwrap] Running Blender in background mode...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the unwrapped mesh
            print(f"[BlenderUVUnwrap] Loading unwrapped mesh...")
            unwrapped = trimesh.load(output_path, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(unwrapped, trimesh.Scene):
                unwrapped = unwrapped.dump(concatenate=True)

            # Preserve metadata
            unwrapped.metadata = mesh.metadata.copy()
            unwrapped.metadata['uv_unwrap'] = {
                'algorithm': 'blender_smart_uv',
                'angle_limit': angle_limit,
                'island_margin': island_margin
            }

            print(f"[BlenderUVUnwrap] ✓ Complete: {len(unwrapped.vertices)} vertices, {len(unwrapped.faces)} faces")

            return (unwrapped,)

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class BlenderVoxelRemeshNode:
    """
    Voxel-based remeshing using Blender.

    Creates a new mesh by voxelizing the input mesh and reconstructing
    the surface. Good for creating uniform, watertight meshes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "voxel_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "voxel_remesh"
    CATEGORY = "geompack/blender"

    def voxel_remesh(self, mesh, voxel_size):
        """
        Apply voxel remeshing using Blender.

        Args:
            mesh: Input trimesh.Trimesh object
            voxel_size: Voxel size for remeshing (smaller = higher resolution)

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[BlenderVoxelRemesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[BlenderVoxelRemesh] Voxel size: {voxel_size}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for voxel remeshing
            script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply voxel remesh
obj.data.remesh_voxel_size = {voxel_size}
bpy.ops.object.voxel_remesh()

# Export remeshed object
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

            print(f"[BlenderVoxelRemesh] Running Blender in background mode...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the remeshed mesh
            print(f"[BlenderVoxelRemesh] Loading remeshed mesh...")
            remeshed = trimesh.load(output_path, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(remeshed, trimesh.Scene):
                remeshed = remeshed.dump(concatenate=True)

            # Preserve metadata
            remeshed.metadata = mesh.metadata.copy()
            remeshed.metadata['remeshing'] = {
                'algorithm': 'blender_voxel',
                'voxel_size': voxel_size,
                'original_vertices': len(mesh.vertices),
                'original_faces': len(mesh.faces),
                'remeshed_vertices': len(remeshed.vertices),
                'remeshed_faces': len(remeshed.faces)
            }

            vertex_change = len(remeshed.vertices) - len(mesh.vertices)
            face_change = len(remeshed.faces) - len(mesh.faces)

            print(f"[BlenderVoxelRemesh] ✓ Complete:")
            print(f"[BlenderVoxelRemesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed.vertices)} ({vertex_change:+d})")
            print(f"[BlenderVoxelRemesh]   Faces:    {len(mesh.faces)} -> {len(remeshed.faces)} ({face_change:+d})")

            return (remeshed,)

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class PreviewMeshNode:
    """
    Preview mesh with interactive 3D viewer.

    Displays mesh in an interactive Three.js viewer with orbit controls.
    Allows rotating, panning, and zooming to inspect the mesh geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh"
    CATEGORY = "geompack/visualization"

    def preview_mesh(self, mesh):
        """
        Export mesh to GLB and prepare for 3D preview.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        import uuid

        print(f"[PreviewMesh] Preparing preview: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to GLB (best format for Three.js)
        try:
            mesh.export(filepath, file_type='glb')
            print(f"[PreviewMesh] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMesh] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            mesh.export(filepath, file_type='obj')
            print(f"[PreviewMesh] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = mesh.bounds
        extents = mesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        return {
            "ui": {
                "mesh_file": [filename],
                "vertex_count": [len(mesh.vertices)],
                "face_count": [len(mesh.faces)],
                "bounds_min": [bounds[0].tolist()],
                "bounds_max": [bounds[1].tolist()],
                "extents": [extents.tolist()],
                "max_extent": [float(max_extent)],
            }
        }


class PreviewMeshVTKNode:
    """
    Preview mesh with VTK.js scientific visualization viewer.

    Displays mesh in an interactive VTK.js viewer with trackball controls.
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk(self, mesh):
        """
        Export mesh to STL and prepare for VTK.js preview.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        import uuid

        print(f"[PreviewMeshVTK] Preparing preview: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.stl"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to STL (native format for VTK.js)
        try:
            mesh.export(filepath, file_type='stl')
            print(f"[PreviewMeshVTK] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTK] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.stl', '.obj')
            filepath = filepath.replace('.stl', '.obj')
            mesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTK] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = mesh.bounds
        extents = mesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        return {
            "ui": {
                "mesh_file": [filename],
                "vertex_count": [len(mesh.vertices)],
                "face_count": [len(mesh.faces)],
                "bounds_min": [bounds[0].tolist()],
                "bounds_max": [bounds[1].tolist()],
                "extents": [extents.tolist()],
                "max_extent": [float(max_extent)],
            }
        }


class CenterMeshNode:
    """
    Center mesh at origin (0, 0, 0).

    Uses bounding box center to translate the mesh so its center
    is at the world origin. Useful for preparing meshes for export
    or ensuring consistent positioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("centered_mesh",)
    FUNCTION = "center_mesh"
    CATEGORY = "geompack/transforms"

    def center_mesh(self, mesh):
        """
        Center mesh at origin using bounding box center.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            tuple: (centered_trimesh.Trimesh,)
        """
        print(f"[CenterMesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Calculate bounding box center
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0

        print(f"[CenterMesh] Original center: [{bounds_center[0]:.3f}, {bounds_center[1]:.3f}, {bounds_center[2]:.3f}]")

        # Apply translation to center at origin
        mesh_centered = mesh.copy()
        mesh_centered.apply_translation(-bounds_center)

        # Verify centering
        new_center = (mesh_centered.bounds[0] + mesh_centered.bounds[1]) / 2.0
        print(f"[CenterMesh] New center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]")

        # Preserve metadata
        mesh_centered.metadata = mesh.metadata.copy()
        mesh_centered.metadata['centered'] = True
        mesh_centered.metadata['original_center'] = bounds_center.tolist()

        return (mesh_centered,)


class BlenderQuadriflowRemeshNode:
    """
    Quadriflow remeshing using Blender.

    Creates a quad-dominant mesh with good topology. Better for animation
    and subdivision surfaces than triangle-based remeshing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 100000,
                    "step": 100
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "quadriflow_remesh"
    CATEGORY = "geompack/blender"

    def quadriflow_remesh(self, mesh, target_face_count):
        """
        Apply Quadriflow remeshing using Blender.

        Args:
            mesh: Input trimesh.Trimesh object
            target_face_count: Target number of faces in output mesh

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[BlenderQuadriflow] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[BlenderQuadriflow] Target face count: {target_face_count}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for Quadriflow remeshing
            script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply Quadriflow remesh
# Note: Different Blender versions have different parameters
# Using minimal set for maximum compatibility
bpy.ops.object.quadriflow_remesh(
    use_mesh_symmetry=False,
    use_preserve_sharp=False,
    use_preserve_boundary=False,
    smooth_normals=False,
    mode='FACES',
    target_faces={target_face_count},
    seed=0
)

# Export remeshed object
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

            print(f"[BlenderQuadriflow] Running Blender in background mode...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the remeshed mesh
            print(f"[BlenderQuadriflow] Loading remeshed mesh...")
            remeshed = trimesh.load(output_path, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(remeshed, trimesh.Scene):
                remeshed = remeshed.dump(concatenate=True)

            # Quadriflow produces quads, but trimesh will triangulate them
            # Preserve metadata
            remeshed.metadata = mesh.metadata.copy()
            remeshed.metadata['remeshing'] = {
                'algorithm': 'blender_quadriflow',
                'target_face_count': target_face_count,
                'original_vertices': len(mesh.vertices),
                'original_faces': len(mesh.faces),
                'remeshed_vertices': len(remeshed.vertices),
                'remeshed_faces': len(remeshed.faces)
            }

            vertex_change = len(remeshed.vertices) - len(mesh.vertices)
            face_change = len(remeshed.faces) - len(mesh.faces)

            print(f"[BlenderQuadriflow] ✓ Complete:")
            print(f"[BlenderQuadriflow]   Vertices: {len(mesh.vertices)} -> {len(remeshed.vertices)} ({vertex_change:+d})")
            print(f"[BlenderQuadriflow]   Faces:    {len(mesh.faces)} -> {len(remeshed.faces)} ({face_change:+d})")

            return (remeshed,)

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Node class mappings - this dictionary maps internal node names to classes
NODE_CLASS_MAPPINGS = {
    "GeomPackExampleNode": ExampleLibiglNode,
    "GeomPackMeshInfo": MeshInfoNode,
    "GeomPackLoadMesh": LoadMesh,
    "GeomPackSaveMesh": SaveMesh,
    "GeomPackCreatePrimitive": CreatePrimitive,
    "GeomPackCGALRemesh": CGALRemeshNode,
    "GeomPackCenterMesh": CenterMeshNode,
    "GeomPackPreviewMesh": PreviewMeshNode,
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
    "GeomPackBlenderUVUnwrap": BlenderUVUnwrapNode,
    "GeomPackBlenderVoxelRemesh": BlenderVoxelRemeshNode,
    "GeomPackBlenderQuadriflowRemesh": BlenderQuadriflowRemeshNode,
}

# Display name mappings - these are the names shown in ComfyUI's node browser
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExampleNode": "Example Node",
    "GeomPackMeshInfo": "Mesh Info",
    "GeomPackLoadMesh": "Load Mesh",
    "GeomPackSaveMesh": "Save Mesh",
    "GeomPackCreatePrimitive": "Create Primitive",
    "GeomPackCGALRemesh": "CGAL Remesh (Isotropic)",
    "GeomPackCenterMesh": "Center Mesh",
    "GeomPackPreviewMesh": "Preview Mesh (3D)",
    "GeomPackPreviewMeshVTK": "Preview Mesh (VTK)",
    "GeomPackBlenderUVUnwrap": "Blender UV Unwrap",
    "GeomPackBlenderVoxelRemesh": "Blender Voxel Remesh",
    "GeomPackBlenderQuadriflowRemesh": "Blender Quadriflow Remesh",
}
