# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PMP uniform isotropic remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshPMPUniformNode(io.ComfyNode):
    """PMP uniform isotropic remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_PMPUniform",
            display_name="Remesh PMP Uniform (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("pmp_edge_length", default=1.0, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Target edge length for uniform remeshing."),
                io.Int.Input("pmp_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                io.Combo.Input("pmp_use_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto the input surface after each iteration."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, pmp_edge_length=1.0, pmp_iterations=10, pmp_use_projection="true"):
        import pypmp

        log.info("Backend: pmp_uniform")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: edge_length=%s, iterations=%d, use_projection=%s",
                 pmp_edge_length, pmp_iterations, pmp_use_projection)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pypmp.remesh_uniform(
            V, F, pmp_edge_length,
            iterations=pmp_iterations,
            use_projection=(pmp_use_projection == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pmp_uniform',
            'edge_length': pmp_edge_length,
            'iterations': pmp_iterations,
            'use_projection': pmp_use_projection == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (PMP Uniform): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"edge={pmp_edge_length}, iter={pmp_iterations}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_PMPUniform": RemeshPMPUniformNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_PMPUniform": "Remesh PMP Uniform (backend)"}
