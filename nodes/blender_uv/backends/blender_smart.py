# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Blender Smart UV Project backend node."""

import logging
import math

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

from ._helpers import _extract_uvs_from_blender_mesh

log = logging.getLogger("geometrypack")


class UVBlenderSmartNode(io.ComfyNode):
    """Blender Smart UV Project backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_BlenderSmart",
            display_name="UV Blender Smart (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("angle_limit", default=66.0, min=1.0, max=89.0, step=1.0),
                io.Float.Input("island_margin", default=0.02, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, angle_limit=66.0, island_margin=0.02, scale_to_bounds="true"):
        """Blender Smart UV Project using bpy."""
        import bpy
        import bmesh

        log.info("Backend: blender_smart")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: angle_limit=%s, island_margin=%s, scale_to_bounds=%s",
                 angle_limit, island_margin, scale_to_bounds)

        angle_limit_rad = math.radians(angle_limit)

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

        bpy.ops.uv.smart_project(
            angle_limit=angle_limit_rad,
            island_margin=island_margin,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=(scale_to_bounds == 'true')
        )

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
            'algorithm': 'blender_smart_uv_project',
            'angle_limit': angle_limit,
            'island_margin': island_margin,
            'scale_to_bounds': scale_to_bounds == 'true'
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (Blender Smart UV):

Method: Smart UV Project (bpy)
Angle Limit: {angle_limit}deg
Island Margin: {island_margin}
Scale to Bounds: {scale_to_bounds}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(unwrapped.vertices):,}
  Faces: {len(unwrapped.faces):,}

Automatic seam-based unwrapping with intelligent island creation.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_BlenderSmart": UVBlenderSmartNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_BlenderSmart": "UV Blender Smart (backend)"}
