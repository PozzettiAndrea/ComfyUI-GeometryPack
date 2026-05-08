# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geogram CVT isotropic remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshGeogramSmoothNode(io.ComfyNode):
    """Geogram CVT isotropic remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_GeogramSmooth",
            display_name="Remesh Geogram Smooth (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("nb_points", default=5000, min=0, max=1000000, step=100, tooltip="Target number of output vertices. 0 = same count as input."),
                io.Int.Input("nb_lloyd_iter", default=5, min=1, max=50, step=1, tooltip="Lloyd relaxation iterations (initial uniform distribution)."),
                io.Int.Input("nb_newton_iter", default=30, min=1, max=100, step=1, tooltip="Newton optimization iterations (refines point placement)."),
                io.Int.Input("newton_m", default=7, min=1, max=20, step=1, tooltip="Number of L-BFGS Hessian evaluations."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, nb_points=5000, nb_lloyd_iter=5, nb_newton_iter=30, newton_m=7):
        import pygeogram

        log.info("Backend: geogram_smooth")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: nb_points=%s, nb_lloyd_iter=%s, nb_newton_iter=%s, newton_m=%s",
                 f"{nb_points:,}", nb_lloyd_iter, nb_newton_iter, newton_m)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pygeogram.remesh_smooth(
            V, F, nb_points,
            nb_lloyd_iter=nb_lloyd_iter,
            nb_newton_iter=nb_newton_iter,
            newton_m=newton_m,
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_smooth',
            'nb_points': nb_points,
            'nb_lloyd_iter': nb_lloyd_iter,
            'nb_newton_iter': nb_newton_iter,
            'newton_m': newton_m,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Geogram CVT Smooth): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"nb_points={nb_points:,}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_GeogramSmooth": RemeshGeogramSmoothNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_GeogramSmooth": "Remesh Geogram Smooth (backend)"}
