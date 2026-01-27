# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Boolean Operations Nodes - CSG operations on meshes
"""

import numpy as np
import trimesh as trimesh_module


def _bpy_boolean_operation(vertices_a, faces_a, vertices_b, faces_b, operation):
    """Blender boolean operation with EXACT solver using bpy."""
    from .._utils import setup_bpy_dll_path
    setup_bpy_dll_path()
    import bpy

    # Create mesh A
    mesh_a = bpy.data.meshes.new("MeshA")
    obj_a = bpy.data.objects.new("ObjectA", mesh_a)
    bpy.context.collection.objects.link(obj_a)
    mesh_a.from_pydata(vertices_a.tolist(), [], faces_a.tolist())
    mesh_a.update()

    # Create mesh B
    mesh_b = bpy.data.meshes.new("MeshB")
    obj_b = bpy.data.objects.new("ObjectB", mesh_b)
    bpy.context.collection.objects.link(obj_b)
    mesh_b.from_pydata(vertices_b.tolist(), [], faces_b.tolist())
    mesh_b.update()

    # Select A as active
    bpy.ops.object.select_all(action='DESELECT')
    obj_a.select_set(True)
    bpy.context.view_layer.objects.active = obj_a

    # Add boolean modifier
    bool_mod = obj_a.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = operation
    bool_mod.object = obj_b
    bool_mod.solver = 'EXACT'

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier="Boolean")

    mesh_a = obj_a.data
    result_vertices = [list(v.co) for v in mesh_a.vertices]
    result_faces = [list(p.vertices) for p in mesh_a.polygons]

    # Cleanup
    bpy.data.objects.remove(obj_b, do_unlink=True)
    bpy.data.meshes.remove(mesh_b)
    bpy.data.objects.remove(obj_a, do_unlink=True)
    bpy.data.meshes.remove(mesh_a)

    return {'vertices': result_vertices, 'faces': result_faces}


class BooleanOpNode:
    """
    Boolean Operations - Union, Difference, and Intersection of meshes.

    Performs Constructive Solid Geometry (CSG) operations:
    - union: Combine two meshes into one
    - difference: Subtract mesh_b from mesh_a
    - intersection: Keep only overlapping parts

    Uses libigl with CGAL backend for robust boolean operations.
    Fallback to Blender if CGAL is not available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("TRIMESH",),
                "mesh_b": ("TRIMESH",),
                "operation": (["union", "difference", "intersection"],),
            },
            "optional": {
                "engine": (["auto", "libigl_cgal", "blender"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("result_mesh", "info")
    FUNCTION = "boolean_op"
    CATEGORY = "geompack/boolean"

    def boolean_op(self, mesh_a, mesh_b, operation, engine="auto"):
        """
        Perform boolean operation on two meshes.

        Args:
            mesh_a: First mesh (base mesh for difference)
            mesh_b: Second mesh (subtracted mesh for difference)
            operation: Boolean operation type
            engine: Backend to use (auto, libigl_cgal, blender)

        Returns:
            tuple: (result_mesh, info_string)
        """
        print(f"[Boolean] Mesh A: {len(mesh_a.vertices)} vertices, {len(mesh_a.faces)} faces")
        print(f"[Boolean] Mesh B: {len(mesh_b.vertices)} vertices, {len(mesh_b.faces)} faces")
        print(f"[Boolean] Operation: {operation}, Engine: {engine}")

        # Try libigl+CGAL backend first
        if engine in ["auto", "libigl_cgal"]:
            result, info = self._try_libigl_cgal(mesh_a, mesh_b, operation)
            if result is not None:
                return (result, info)
            if engine == "libigl_cgal":
                raise RuntimeError("libigl+CGAL backend failed and was explicitly requested")

        # Fallback to Blender
        if engine in ["auto", "blender"]:
            result, info = self._try_blender(mesh_a, mesh_b, operation)
            if result is not None:
                return (result, info)

        raise RuntimeError(f"Boolean operation failed with all available backends")

    def _try_libigl_cgal(self, mesh_a, mesh_b, operation):
        """Try boolean operation using libigl with CGAL backend."""
        try:
            import igl.copyleft.cgal as cgal
            print(f"[Boolean] Attempting libigl+CGAL backend...")

            # Convert trimesh to numpy arrays
            VA = np.asarray(mesh_a.vertices, dtype=np.float64)
            FA = np.asarray(mesh_a.faces, dtype=np.int64)
            VB = np.asarray(mesh_b.vertices, dtype=np.float64)
            FB = np.asarray(mesh_b.faces, dtype=np.int64)

            # Map operation to igl type_str
            # mesh_boolean accepts string: "union", "intersection", "difference"
            op_map = {
                "union": "union",
                "difference": "difference",
                "intersection": "intersection"
            }

            if operation not in op_map:
                raise ValueError(f"Unknown operation: {operation}")

            # Perform boolean operation using CGAL
            VC, FC, J = cgal.mesh_boolean(VA, FA, VB, FB, op_map[operation])

            # Create result trimesh
            result = trimesh_module.Trimesh(vertices=VC, faces=FC, process=False)

            # Preserve metadata from mesh_a
            result.metadata = mesh_a.metadata.copy()
            result.metadata['boolean'] = {
                'operation': operation,
                'engine': 'libigl_cgal',
                'mesh_a_vertices': len(mesh_a.vertices),
                'mesh_a_faces': len(mesh_a.faces),
                'mesh_b_vertices': len(mesh_b.vertices),
                'mesh_b_faces': len(mesh_b.faces),
                'result_vertices': len(result.vertices),
                'result_faces': len(result.faces)
            }

            info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: libigl + CGAL

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

            print(f"[Boolean] libigl+CGAL success: {len(result.vertices)} vertices, {len(result.faces)} faces")
            return result, info

        except Exception as e:
            print(f"[Boolean] libigl+CGAL backend failed: {e}")
            return None, str(e)

    def _try_blender(self, mesh_a, mesh_b, operation):
        """Try boolean operation using Blender via bpy."""
        try:
            print(f"[Boolean] Attempting Blender backend...")

            # Map operation to Blender modifier type
            blender_op = {
                "union": "UNION",
                "difference": "DIFFERENCE",
                "intersection": "INTERSECT"
            }[operation]

            result_data = _bpy_boolean_operation(
                vertices_a=np.asarray(mesh_a.vertices, dtype=np.float32),
                faces_a=np.asarray(mesh_a.faces, dtype=np.int32),
                vertices_b=np.asarray(mesh_b.vertices, dtype=np.float32),
                faces_b=np.asarray(mesh_b.faces, dtype=np.int32),
                operation=blender_op
            )

            result = trimesh_module.Trimesh(
                vertices=np.array(result_data['vertices'], dtype=np.float32),
                faces=np.array(result_data['faces'], dtype=np.int32),
                process=False
            )

            # Preserve metadata
            result.metadata = mesh_a.metadata.copy()
            result.metadata['boolean'] = {
                'operation': operation,
                'engine': 'blender_bpy',
                'mesh_a_vertices': len(mesh_a.vertices),
                'mesh_a_faces': len(mesh_a.faces),
                'mesh_b_vertices': len(mesh_b.vertices),
                'mesh_b_faces': len(mesh_b.faces),
                'result_vertices': len(result.vertices),
                'result_faces': len(result.faces)
            }

            info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: blender bpy (EXACT solver)

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

            print(f"[Boolean] Blender success: {len(result.vertices)} vertices, {len(result.faces)} faces")
            return result, info

        except Exception as e:
            print(f"[Boolean] Blender backend failed: {e}")
            return None, str(e)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackBooleanOp": BooleanOpNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBooleanOp": "Boolean Operation",
}
