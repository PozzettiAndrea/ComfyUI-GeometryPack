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
                io.Int.Input("octree_depth", default=6, min=1, max=12, step=1, tooltip="Octree resolution -- the detail knob. Power of 2: each +1 roughly QUADRUPLES face count and halves voxel size. 6 is a sane start; 8-9 is high detail; 10+ can be very heavy."),
                io.Float.Input("scale", default=0.9, min=0.0, max=0.99, step=0.05, display_mode="number", tooltip="Octree fit relative to the bounding box (0-0.99). Higher = grid hugs the mesh tighter = finer effective resolution; too close to 1.0 can clip the outer shell. 0.9 default."),
                io.Float.Input("sharpness", default=1.0, min=0.0, max=2.0, step=0.1, display_mode="number", tooltip="How aggressively dual-contouring snaps to sharp edges/corners. Higher = crisper edges but can spike on noisy input; lower = rounder. 0-2 is Blender's normal slider range (1.0 default); capped at 2 here since higher just over-sharpens."),
                io.Combo.Input("remove_disconnected", options=["true", "false"], default="true", tooltip="Delete small disconnected (floating) pieces after remeshing. ON by default (matches Blender)."),
                io.Float.Input("disconnected_threshold", default=1.0, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Size cutoff for removal, relative to the largest component. Higher = remove more aggressively (1.0 ~ keep only the main body); lower = keep more; 0 = keep everything. Only used when remove_disconnected is on."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, octree_depth=6, scale=0.9, sharpness=1.0,
                remove_disconnected="true", disconnected_threshold=1.0):
        import bpy

        log.info("Backend: blender_sharp")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: octree_depth=%d, scale=%s, sharpness=%s, remove_disconnected=%s, threshold=%s",
                 octree_depth, scale, sharpness, remove_disconnected, disconnected_threshold)

        obj, mesh = _bpy_setup_object(
            np.asarray(trimesh.vertices, dtype=np.float32),
            np.asarray(trimesh.faces, dtype=np.int32)
        )
        mod = obj.modifiers.new(name="Remesh", type='REMESH')
        mod.mode = 'SHARP'
        mod.octree_depth = octree_depth
        mod.scale = scale
        mod.sharpness = sharpness
        mod.use_remove_disconnected = (remove_disconnected == "true")
        mod.threshold = disconnected_threshold
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
            'remove_disconnected': remove_disconnected == "true", 'disconnected_threshold': disconnected_threshold,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Blender Sharp): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"depth={octree_depth}, sharpness={sharpness}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_BlenderSharp": RemeshBlenderSharpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_BlenderSharp": "Remesh Blender Sharp (backend)"}
