# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Blender EXACT solver boolean operations backend node."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _bpy_boolean_operation(vertices_a, faces_a, vertices_b, faces_b, operation):
    """Blender boolean operation with EXACT solver using bpy."""
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

    # Triangulate to ensure uniform face arrays (Blender may produce n-gons)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh_a = obj_a.data
    result_vertices = [list(v.co) for v in mesh_a.vertices]
    result_faces = [list(p.vertices) for p in mesh_a.polygons]

    # Cleanup
    bpy.data.objects.remove(obj_b, do_unlink=True)
    bpy.data.meshes.remove(mesh_b)
    bpy.data.objects.remove(obj_a, do_unlink=True)
    bpy.data.meshes.remove(mesh_a)

    return {'vertices': result_vertices, 'faces': result_faces}


class BooleanBlenderExactNode(io.ComfyNode):
    """Blender EXACT solver boolean operations backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBoolean_BlenderExact",
            display_name="Boolean Blender Exact (backend)",
            category="geompack/boolean",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_a"),
                io.Custom("TRIMESH").Input("mesh_b"),
                io.Combo.Input("operation", options=["union", "difference", "intersection"]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="result_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh_a, mesh_b, operation="union"):
        log.info("Backend: blender_exact")
        log.info("Mesh A: %d vertices, %d faces", len(mesh_a.vertices), len(mesh_a.faces))
        log.info("Mesh B: %d vertices, %d faces", len(mesh_b.vertices), len(mesh_b.faces))
        log.info("Operation: %s", operation)

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
Engine: Blender bpy (EXACT solver)

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

        log.info("Success: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackBoolean_BlenderExact": BooleanBlenderExactNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackBoolean_BlenderExact": "Boolean Blender Exact (backend)"}
