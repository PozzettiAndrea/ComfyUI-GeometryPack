# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""QuadriFlow quad remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshQuadriFlowNode(io.ComfyNode):
    """QuadriFlow quad remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_QuadriFlow",
            display_name="Remesh QuadriFlow (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=5000, min=100, max=5000000, step=100, tooltip="Target number of output faces. Creates quad-dominant mesh."),
                io.Combo.Input("preserve_sharp", options=["true", "false"], default="false", tooltip="Preserve sharp edges during remeshing."),
                io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Preserve mesh boundary edges during remeshing."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=5000, preserve_sharp="false", preserve_boundary="true"):
        from pyquadriflow.quadriflow import quadriflow_remesh

        log.info("Backend: quadriflow")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: target_face_count=%s, preserve_sharp=%s, preserve_boundary=%s",
                 f"{target_face_count:,}", preserve_sharp, preserve_boundary)

        V = np.asarray(trimesh.vertices, dtype=np.float64)
        F = np.asarray(trimesh.faces, dtype=np.int32)

        out_vertices, out_faces = quadriflow_remesh(
            V, F, target_face_count,
            preserve_sharp=preserve_sharp == "true",
            preserve_boundary=preserve_boundary == "true",
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.asarray(out_vertices, dtype=np.float32),
            faces=np.asarray(out_faces, dtype=np.int32),
            process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadriflow',
            'target_face_count': target_face_count,
            'preserve_sharp': preserve_sharp == "true",
            'preserve_boundary': preserve_boundary == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (QuadriFlow): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"target_faces={target_face_count:,}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_QuadriFlow": RemeshQuadriFlowNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_QuadriFlow": "Remesh QuadriFlow (backend)"}
