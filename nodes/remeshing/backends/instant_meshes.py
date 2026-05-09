# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Instant Meshes field-aligned remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshInstantMeshesNode(io.ComfyNode):
    """Instant Meshes field-aligned remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_InstantMeshes",
            display_name="Remesh Instant Meshes (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_vertex_count", default=5000, min=100, max=1000000, step=100, tooltip="Target vertex count. Creates field-aligned quad-dominant mesh."),
                io.Combo.Input("deterministic", options=["true", "false"], default="true", tooltip="Use deterministic algorithm for reproducible results."),
                io.Float.Input("crease_angle", default=0.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold for preserving sharp/crease edges. 0 = no crease preservation."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_vertex_count=5000, deterministic="true", crease_angle=0.0):
        import pynanoinstantmeshes as pynano

        log.info("Backend: instant_meshes")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: target_vertex_count=%s, deterministic=%s, crease_angle=%s",
                 f"{target_vertex_count:,}", deterministic, crease_angle)

        V = trimesh.vertices.astype(np.float32)
        F = trimesh.faces.astype(np.uint32)

        V_out, F_out = pynano.remesh(
            V, F,
            vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            creaseAngle=crease_angle
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'instant_meshes',
            'target_vertex_count': target_vertex_count,
            'deterministic': deterministic == "true",
            'crease_angle': crease_angle,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Instant Meshes): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"target_verts={target_vertex_count:,}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_InstantMeshes": RemeshInstantMeshesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_InstantMeshes": "Remesh Instant Meshes (backend)"}
