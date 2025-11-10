"""
Remeshing Nodes - Mesh remeshing and optimization algorithms
"""

import numpy as np
import trimesh
import os
import subprocess
import tempfile
import shutil

from . import mesh_utils


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


class PyMeshLabRemeshNode:
    """
    PyMeshLab Isotropic Remeshing - Create uniform triangle meshes.

    Uses PyMeshLab's implementation of isotropic remeshing.
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
    CATEGORY = "geompack/pymeshlab"

    def remesh(self, mesh, target_edge_length, iterations):
        """
        Apply PyMeshLab isotropic remeshing.

        Args:
            mesh: Input trimesh.Trimesh object
            target_edge_length: Target edge length for remeshed triangles
            iterations: Number of remeshing iterations

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[PyMeshLabRemesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[PyMeshLabRemesh] Target edge length: {target_edge_length}, Iterations: {iterations}")

        remeshed_mesh, error = mesh_utils.pymeshlab_isotropic_remesh(
            mesh,
            target_edge_length,
            iterations
        )

        if remeshed_mesh is None:
            raise ValueError(f"Remeshing failed: {error}")

        print(f"[PyMeshLabRemesh] Output: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

        return (remeshed_mesh,)


class CGALIsotropicRemeshNode:
    """
    CGAL Isotropic Remeshing - Create uniform triangle meshes using CGAL.

    Uses CGAL's high-quality isotropic remeshing algorithm to create meshes
    with uniform triangle sizes and improved quality. Preserves volume and
    surface features while creating more regular triangulations.

    Requires the CGAL Python package: pip install cgal
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
                "protect_boundaries": (["true", "false"], {
                    "default": "true"
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "remesh"
    CATEGORY = "geompack/cgal"

    def remesh(self, mesh, target_edge_length, iterations, protect_boundaries="true"):
        """
        Apply CGAL isotropic remeshing.

        Args:
            mesh: Input trimesh.Trimesh object
            target_edge_length: Target edge length for remeshed triangles
            iterations: Number of remeshing iterations (1-20)
            protect_boundaries: Whether to preserve boundary edges ("true" or "false")

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[CGALRemesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[CGALRemesh] Target edge length: {target_edge_length}, Iterations: {iterations}")
        print(f"[CGALRemesh] Protect boundaries: {protect_boundaries}")

        protect = (protect_boundaries == "true")

        remeshed_mesh, error = mesh_utils.cgal_isotropic_remesh(
            mesh,
            target_edge_length,
            iterations,
            protect
        )

        if remeshed_mesh is None:
            raise ValueError(f"CGAL remeshing failed: {error}")

        print(f"[CGALRemesh] Output: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

        return (remeshed_mesh,)


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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackPyMeshLabRemesh": PyMeshLabRemeshNode,
    "GeomPackCGALIsotropicRemesh": CGALIsotropicRemeshNode,
    "GeomPackBlenderVoxelRemesh": BlenderVoxelRemeshNode,
    "GeomPackBlenderQuadriflowRemesh": BlenderQuadriflowRemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPyMeshLabRemesh": "PyMeshLab Remesh (Isotropic)",
    "GeomPackCGALIsotropicRemesh": "CGAL Isotropic Remesh",
    "GeomPackBlenderVoxelRemesh": "Blender Voxel Remesh",
    "GeomPackBlenderQuadriflowRemesh": "Blender Quadriflow Remesh",
}
