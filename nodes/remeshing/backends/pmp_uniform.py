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
                io.Float.Input("pmp_edge_length", default=1.0, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Target edge length for uniform remeshing. Used only when target_vertices is 0."),
                io.Int.Input("pmp_target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertices (0 = off, use edge_length). Back-solves the edge length from the input mesh area; overrides edge_length. Approximate."),
                io.Int.Input("pmp_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                io.Combo.Input("pmp_use_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto the input surface after each iteration."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, pmp_edge_length=1.0, pmp_target_vertices=0,
                pmp_iterations=10, pmp_use_projection="true"):
        import math
        import pypmp

        # Resolve edge length from a target vertex count if given (0 = off). For an
        # isotropic TRIANGLE mesh of area A with edge e: #verts ~= 2A/(sqrt(3)*e^2),
        # so e = sqrt(2A / (sqrt(3) * V)). Approximate.
        edge = float(pmp_edge_length)
        note = f"edge={edge:.4g}"
        tgt = int(pmp_target_vertices or 0)
        if tgt > 0:
            try:
                area = float(trimesh.area)
            except Exception:
                area = 0.0
            if area > 0.0:
                edge = math.sqrt(2.0 * area / (math.sqrt(3.0) * tgt))
                note = f"target_vertices={tgt:,} -> edge~={edge:.4g} (area={area:.4g})"

        log.info("Backend: pmp_uniform")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: %s, iterations=%d, use_projection=%s",
                 note, pmp_iterations, pmp_use_projection)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pypmp.remesh_uniform(
            V, F, edge,
            iterations=pmp_iterations,
            use_projection=(pmp_use_projection == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pmp_uniform',
            'edge_length': edge,
            'target_vertices': tgt,
            'iterations': pmp_iterations,
            'use_projection': pmp_use_projection == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (PMP Uniform): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"{note}, iter={pmp_iterations}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_PMPUniform": RemeshPMPUniformNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_PMPUniform": "Remesh PMP Uniform (backend)"}
