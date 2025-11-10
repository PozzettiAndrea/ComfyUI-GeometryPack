"""
UV Mapping Nodes - UV unwrapping and parameterization
"""

import numpy as np
import trimesh
import os
import subprocess
import tempfile
import shutil


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


class XAtlasUVUnwrapNode:
    """
    UV Unwrap mesh using xatlas library.

    Fast, automatic UV unwrapping optimized for lightmaps and texture atlasing.
    No Blender dependency required. Uses the same algorithm as Blender 3.6+
    for UV packing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/uv"

    def uv_unwrap(self, mesh):
        """
        UV unwrap mesh using xatlas.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        try:
            import xatlas
        except ImportError:
            raise ImportError(
                "xatlas not installed. Install with: pip install xatlas\n"
                "This is required for fast UV unwrapping without Blender."
            )

        print(f"[XAtlasUVUnwrap] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Parametrize with xatlas
        vmapping, indices, uvs = xatlas.parametrize(
            mesh.vertices,
            mesh.faces
        )

        # Create new mesh with UV-split vertices
        new_vertices = mesh.vertices[vmapping]

        # Create trimesh with UV coordinates
        unwrapped = trimesh.Trimesh(
            vertices=new_vertices,
            faces=indices,
            process=False
        )

        # Store UV coordinates in visual
        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uvs)

        # Preserve metadata
        unwrapped.metadata = mesh.metadata.copy()
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'xatlas',
            'original_vertices': len(mesh.vertices),
            'unwrapped_vertices': len(new_vertices),
            'vertex_duplication_ratio': len(new_vertices) / len(mesh.vertices)
        }

        print(f"[XAtlasUVUnwrap] Output: {len(unwrapped.vertices)} vertices, {len(unwrapped.faces)} faces")
        print(f"[XAtlasUVUnwrap] Vertex duplication: {len(new_vertices)/len(mesh.vertices):.2f}x")

        return (unwrapped,)


class LibiglLSCMNode:
    """
    LSCM UV Parameterization using libigl.

    Least Squares Conformal Maps - minimizes angle distortion.
    Fast, conformal mapping suitable for texturing organic shapes.
    No Blender dependency required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/uv"

    def uv_unwrap(self, mesh):
        """
        LSCM UV parameterization using libigl.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        try:
            import igl
        except ImportError:
            raise ImportError("libigl not installed (should be in requirements.txt)")

        print(f"[LibiglLSCM] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # LSCM requires fixing 2 vertices for unique solution
        # Choose first and last vertex
        v_fixed = np.array([0, len(mesh.vertices)-1], dtype=np.int32)
        uv_fixed = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        # Compute LSCM parameterization
        uv = igl.lscm(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int32),
            v_fixed,
            uv_fixed
        )

        # Normalize UVs to [0, 1] range
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv_range = uv_max - uv_min

        # Avoid division by zero
        uv_range[uv_range < 1e-10] = 1.0

        uv_normalized = (uv - uv_min) / uv_range

        # Create unwrapped mesh (copy original)
        unwrapped = mesh.copy()

        # Store UV coordinates in visual
        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uv_normalized)

        # Add metadata
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'libigl_lscm',
            'conformal': True,
            'angle_preserving': True,
            'fixed_vertices': v_fixed.tolist()
        }

        print(f"[LibiglLSCM] Complete - conformal (angle-preserving) mapping")

        return (unwrapped,)


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


class BlenderCubeProjectionNode:
    """
    UV Cube Projection using Blender.

    Projects mesh onto 6 faces of a cube. Perfect for box-like geometry.
    Creates 6 overlapping UV islands that can be separated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "cube_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/uv"

    def uv_unwrap(self, mesh, cube_size):
        """
        UV cube projection using Blender.

        Args:
            mesh: Input trimesh.Trimesh object
            cube_size: Size of the projection cube

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        print(f"[BlenderCubeProjection] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[BlenderCubeProjection] Cube size: {cube_size}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for cube projection
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

# Switch to edit mode and apply cube projection
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.cube_project(
    cube_size={cube_size},
    correct_aspect=True,
    clip_to_bounds=False,
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

            print(f"[BlenderCubeProjection] Running Blender...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the unwrapped mesh
            unwrapped = trimesh.load(output_path, process=False)

            if isinstance(unwrapped, trimesh.Scene):
                unwrapped = unwrapped.dump(concatenate=True)

            # Preserve metadata
            unwrapped.metadata = mesh.metadata.copy()
            unwrapped.metadata['uv_unwrap'] = {
                'algorithm': 'blender_cube_projection',
                'cube_size': cube_size
            }

            print(f"[BlenderCubeProjection] Complete")

            return (unwrapped,)

        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class BlenderCylinderProjectionNode:
    """
    UV Cylinder Projection using Blender.

    Projects mesh onto a cylinder surface. Perfect for cylindrical objects
    like bottles, columns, pipes, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/uv"

    def uv_unwrap(self, mesh, radius):
        """
        UV cylinder projection using Blender.

        Args:
            mesh: Input trimesh.Trimesh object
            radius: Cylinder radius

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        print(f"[BlenderCylinderProjection] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[BlenderCylinderProjection] Radius: {radius}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for cylinder projection
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

# Switch to edit mode and apply cylinder projection
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.cylinder_project(
    direction='VIEW_ON_EQUATOR',
    align='POLAR_ZX',
    radius={radius},
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

            print(f"[BlenderCylinderProjection] Running Blender...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the unwrapped mesh
            unwrapped = trimesh.load(output_path, process=False)

            if isinstance(unwrapped, trimesh.Scene):
                unwrapped = unwrapped.dump(concatenate=True)

            # Preserve metadata
            unwrapped.metadata = mesh.metadata.copy()
            unwrapped.metadata['uv_unwrap'] = {
                'algorithm': 'blender_cylinder_projection',
                'radius': radius
            }

            print(f"[BlenderCylinderProjection] Complete")

            return (unwrapped,)

        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class BlenderSphereProjectionNode:
    """
    UV Sphere Projection using Blender.

    Projects mesh onto a sphere surface. Perfect for spherical objects
    like planets, balls, eyes, etc. Creates equirectangular projection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("unwrapped_mesh",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "geompack/uv"

    def uv_unwrap(self, mesh):
        """
        UV sphere projection using Blender.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            tuple: (unwrapped_trimesh.Trimesh,)
        """
        print(f"[BlenderSphereProjection] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for sphere projection
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

# Switch to edit mode and apply sphere projection
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.sphere_project(
    direction='VIEW_ON_EQUATOR',
    align='POLAR_ZX',
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

            print(f"[BlenderSphereProjection] Running Blender...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the unwrapped mesh
            unwrapped = trimesh.load(output_path, process=False)

            if isinstance(unwrapped, trimesh.Scene):
                unwrapped = unwrapped.dump(concatenate=True)

            # Preserve metadata
            unwrapped.metadata = mesh.metadata.copy()
            unwrapped.metadata['uv_unwrap'] = {
                'algorithm': 'blender_sphere_projection',
            }

            print(f"[BlenderSphereProjection] Complete")

            return (unwrapped,)

        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackXAtlasUVUnwrap": XAtlasUVUnwrapNode,
    "GeomPackLibiglLSCM": LibiglLSCMNode,
    "GeomPackBlenderUVUnwrap": BlenderUVUnwrapNode,
    "GeomPackBlenderCubeProjection": BlenderCubeProjectionNode,
    "GeomPackBlenderCylinderProjection": BlenderCylinderProjectionNode,
    "GeomPackBlenderSphereProjection": BlenderSphereProjectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackXAtlasUVUnwrap": "xAtlas UV Unwrap",
    "GeomPackLibiglLSCM": "libigl LSCM Unwrap",
    "GeomPackBlenderUVUnwrap": "Blender UV Unwrap",
    "GeomPackBlenderCubeProjection": "Blender Cube Projection",
    "GeomPackBlenderCylinderProjection": "Blender Cylinder Projection",
    "GeomPackBlenderSphereProjection": "Blender Sphere Projection",
}
