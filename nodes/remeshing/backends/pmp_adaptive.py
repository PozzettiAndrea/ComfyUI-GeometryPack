# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PMP curvature-driven adaptive remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshPMPAdaptiveNode(io.ComfyNode):
    """PMP curvature-driven adaptive remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_PMPAdaptive",
            display_name="Remesh PMP Adaptive (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("pmp_min_edge", default=0.1, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Minimum edge length (high-curvature areas)."),
                io.Float.Input("pmp_max_edge", default=2.0, min=0.01, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length (flat areas)."),
                io.Float.Input("pmp_approx_error", default=0.1, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Maximum geometric approximation error."),
                io.Int.Input("pmp_adaptive_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                io.Combo.Input("pmp_adaptive_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto the input surface after each iteration."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, pmp_min_edge=0.1, pmp_max_edge=2.0, pmp_approx_error=0.1,
                pmp_adaptive_iterations=10, pmp_adaptive_projection="true"):
        import pypmp

        log.info("Backend: pmp_adaptive")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: min=%s, max=%s, error=%s, iter=%d, proj=%s",
                 pmp_min_edge, pmp_max_edge, pmp_approx_error, pmp_adaptive_iterations, pmp_adaptive_projection)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pypmp.remesh_adaptive(
            V, F, pmp_min_edge, pmp_max_edge, pmp_approx_error,
            iterations=pmp_adaptive_iterations,
            use_projection=(pmp_adaptive_projection == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pmp_adaptive',
            'min_edge_length': pmp_min_edge,
            'max_edge_length': pmp_max_edge,
            'approx_error': pmp_approx_error,
            'iterations': pmp_adaptive_iterations,
            'use_projection': pmp_adaptive_projection == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (PMP Adaptive): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"min={pmp_min_edge}, max={pmp_max_edge}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_PMPAdaptive": RemeshPMPAdaptiveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_PMPAdaptive": "Remesh PMP Adaptive (backend)"}
