# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Blender sharp remesh modifier backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from .voxel import _bpy_setup_object, _bpy_extract_and_cleanup

log = logging.getLogger("geometrypack")


class RemeshBlenderSharpNode(io.ComfyNode):
    """Blender sharp remesh modifier backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_BlenderSharp",
            display_name="Remesh Blender Sharp (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("octree_depth", default=6, min=1, max=10, step=1, tooltip="Resolution. Higher = more detail, more faces."),
                io.Float.Input("scale", default=0.9, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Ratio of output size to input bounding box."),
                io.Float.Input("sharpness", default=1.0, min=0.0, max=5.0, step=0.1, display_mode="number", tooltip="Edge sharpness for Sharp mode."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, octree_depth=6, scale=0.9, sharpness=1.0):
        import bpy

        log.info("Backend: blender_sharp")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: octree_depth=%d, scale=%s, sharpness=%s", octree_depth, scale, sharpness)

        obj, mesh = _bpy_setup_object(
            np.asarray(trimesh.vertices, dtype=np.float32),
            np.asarray(trimesh.faces, dtype=np.int32)
        )
        mod = obj.modifiers.new(name="Remesh", type='REMESH')
        mod.mode = 'SHARP'
        mod.octree_depth = octree_depth
        mod.scale = scale
        mod.sharpness = sharpness
        bpy.ops.object.modifier_apply(modifier="Remesh")
        result = _bpy_extract_and_cleanup(obj)

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'blender_sharp', 'octree_depth': octree_depth, 'scale': scale, 'sharpness': sharpness,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Blender Sharp): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"depth={octree_depth}, sharpness={sharpness}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_BlenderSharp": RemeshBlenderSharpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_BlenderSharp": "Remesh Blender Sharp (backend)"}
