# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Blender Cube UV projection backend node."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

from ._helpers import _extract_uvs_from_blender_mesh

log = logging.getLogger("geometrypack")


class UVBlenderCubeNode(io.ComfyNode):
    """Blender Cube UV projection backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_BlenderCube",
            display_name="UV Blender Cube (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("cube_size", default=1.0, min=0.1, max=10.0, step=0.1),
                io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, cube_size=1.0, scale_to_bounds="true"):
        """Blender Cube UV Project using bpy."""
        import bpy
        import bmesh

        log.info("Backend: blender_cube")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: cube_size=%s, scale_to_bounds=%s", cube_size, scale_to_bounds)

        vertices = np.asarray(trimesh.vertices, dtype=np.float32)
        faces = np.asarray(trimesh.faces, dtype=np.int32)

        mesh = bpy.data.meshes.new("UVMesh")
        obj = bpy.data.objects.new("UVObject", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mesh)
        for face in bm.faces:
            face.select = True
        bmesh.update_edit_mesh(mesh)

        bpy.ops.uv.cube_project(cube_size=cube_size, scale_to_bounds=(scale_to_bounds == 'true'))

        bpy.ops.object.mode_set(mode='OBJECT')

        result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
        result_verts, result_faces, result_uvs = _extract_uvs_from_blender_mesh(mesh, result_vertices)

        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)

        unwrapped = trimesh_module.Trimesh(
            vertices=np.array(result_verts, dtype=np.float32),
            faces=np.array(result_faces, dtype=np.int32),
            process=False
        )

        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=np.array(result_uvs, dtype=np.float32))

        # Preserve metadata
        unwrapped.metadata = trimesh.metadata.copy()
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'blender_cube_projection',
            'cube_size': cube_size,
            'scale_to_bounds': scale_to_bounds == 'true'
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (Blender Cube):

Method: Cube Projection (bpy)
Cube Size: {cube_size}
Scale to Bounds: {scale_to_bounds}

Vertices: {len(unwrapped.vertices):,}
Faces: {len(unwrapped.faces):,}

Projects mesh onto 6 cube faces.
Best for box-like objects.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_BlenderCube": UVBlenderCubeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_BlenderCube": "UV Blender Cube (backend)"}
